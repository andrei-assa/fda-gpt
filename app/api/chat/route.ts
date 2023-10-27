import {kv} from '@vercel/kv'
import {OpenAIStream, StreamingTextResponse} from 'ai'
import {Configuration, OpenAIApi} from 'openai-edge'

import {auth} from '@/auth'
import {nanoid} from '@/lib/utils'
import {ChatAgentOutputParser} from 'langchain/agents'
import {ChatOpenAI} from "langchain/chat_models/openai"
import {ChatPromptTemplate} from "langchain/prompts";
import {RunnableSequence} from "langchain/schema/runnable";
import {StringOutputParser} from "langchain/schema/output_parser";

export const runtime = 'edge'

class FDAOutputParser extends ChatAgentOutputParser {
    constructor() {
        super()
    }

    parse(text: string): Promise<{
        returnValues: {
            output: string;
        };
        log: string;
        tool?: undefined;
        toolInput?: undefined;
    } | {
        tool: any;
        toolInput: any;
        log: string;
        returnValues?: undefined;
    }> {
        const output = {
            returnValues: {
                output: text.replace("JSON: ", "")
            },
            log: text
        }
        return Promise.resolve(output)
    }
}

const configuration = new Configuration({
    apiKey: process.env.OPENAI_API_KEY
})

type SearchParam = { [key: string]: any };
type SortParam = string;
type FieldToReturn = string;

interface IFDAResponse {
    results: any[];
}

class FDAApi {
    static BASE_URL = "https://api.fda.gov/drug/label.json";

    constructor() {
    }

    private buildQuery(searchParams?: SearchParam[], sort?: SortParam, count?: number, limit?: number, skip?: number): string {
        let query: { [key: string]: any } = {};

        if (searchParams) {
            const searchCriteria: string[] = searchParams.map(searchDict => {
                return Object.entries(searchDict).map(
                    ([field, term]) => `${encodeURIComponent(field)}:${encodeURIComponent(term)}`
                ).join('+AND+');
            });
            query['search'] = searchCriteria.join('+AND+');
        }

        if (sort) {
            query['sort'] = sort;
        }
        if (count) {
            query['count'] = count;
        }
        if (limit) {
            query['limit'] = limit;
        }
        if (skip) {
            query['skip'] = skip;
        }

        const queryString = Object.keys(query).map(key => `${key}=${query[key]}`).join('&');
        return `?${queryString}`;
    }

    private async makeRequest(queryString: string): Promise<any> {
        try {
            const response = await fetch(`${FDAApi.BASE_URL}${queryString}`);
            if (response.ok) {
                return await response.json();
            } else {
                throw new Error(`HTTP Error: ${response.status}`);
            }
        } catch (error) {
            throw error;
        }
    }

    async searchDrug(searchParams: SearchParam[], sort?: SortParam, count?: number, limit?: number, skip?: number): Promise<any> {
        const query = this.buildQuery(searchParams, sort, count, limit, skip);
        return this.makeRequest(query);
    }
}

function extractFieldsFromResults(results: any[], fieldsToReturn: FieldToReturn[]): any[] {
    return results.map(result => {
        const extracted: { [key: string]: any } = {};
        fieldsToReturn.forEach(field => {
            extracted[field] = result[field];
        });
        return extracted;
    });
}

async function FDATool(searchParams: SearchParam[], fieldsToReturn: FieldToReturn[], limit: number = 20): Promise<string> {
    const search = new FDAApi();
    try {
        const response: IFDAResponse = await search.searchDrug(searchParams, undefined, undefined, limit);
        const results = response.results;
        const returnedResults = extractFieldsFromResults(results, fieldsToReturn);
        return JSON.stringify(returnedResults);
    } catch (error) {
        throw error;
    }
}

const systemMessageFDA = `
Translate the user's question into a query for the FDA API, using these examples for reference:

Example Question: "What are the warnings associated with Xarelto?"
JSON: {{"search_params": [{{"openfda.brand_name": "xarelto"}}], "fields_to_return": ["warnings"], "limit": 10}}

Example Question: "What should I do if I overdose on acetaminophen?"
JSON: {{"search_params": [{{"openfda.generic_name": "acetaminophen"}}], "fields_to_return": ["overdosage"], "limit": 10}}

Example Question: "Can you provide a description of Xarelto?"
JSON: {{"search_params": [{{"openfda.brand_name": "xarelto"}}], "fields_to_return": ["description"], "limit": 10}}

Example Question: "What are the side effects of Xarelto?"
JSON: {{"search_params": [{{"openfda.brand_name": "xarelto"}}], "fields_to_return": ["adverse_reactions"], "limit": 10}}

Example Question: "How should Xarelto be administered?"
JSON: {{"search_params": [{{"openfda.brand_name": "xarelto"}}], "fields_to_return": ["dosage_and_administration"], "limit": 10}}

Example Question: "Who manufactures Xarelto?"
JSON: {{"search_params": [{{"openfda.brand_name": "xarelto"}}], "fields_to_return": ["manufacturer_name"], "limit": 10}}

Example Question: "Is there any specific information that patients should know about Xarelto?"
JSON: {{"search_params": [{{"openfda.brand_name": "xarelto"}}], "fields_to_return": ["information_for_patients"], "limit": 10}}

Example Question: "Under what conditions should I stop using Xarelto?"
JSON: {{"search_params": [{{"openfda.brand_name": "xarelto"}}], "fields_to_return": ["questions", "stop_use"], "limit": 10}}

Example Question: "What are the contraindications for Eliquis?"
JSON: {{"search_params": [{{"openfda.brand_name": "eliquis"}}], "fields_to_return": ["contraindications"], "limit": 10}}

Example Question: "What is the abuse potential for Adderall?"
JSON: {{"search_params": [{{"openfda.brand_name": "adderall"}}], "fields_to_return": ["abuse"], "limit": 10}}

Example Question: "What is the abuse potential for Morphine?"
JSON: {{"search_params": [{{"openfda.generic_name": "morphine"}}], "fields_to_return": ["abuse"], "limit": 10}}

Example Question: "Indications for Xarelto?"
JSON: {{"search_params": [{{"openfda.brand_name": "xarelto"}}], "fields_to_return": ["indications_and_usage"], "limit": 10}}
`

const systemMessageSummarize = `
Provide a detailed report to answer the user's question using the following information. Structure the report as follows:
- Summary:
- Key points:
- Details:
`

const apiMessage = `
    {question}
    
    {information}
`

const humanMessage = '{input}'

const openai = new OpenAIApi(configuration)

export async function POST(req: Request) {
    const json = await req.json()
    const {messages, previewToken} = json
    const userId = (await auth())?.user.id

    if (!userId) {
        return new Response('Unauthorized', {
            status: 401
        })
    }

    if (previewToken) {
        configuration.apiKey = previewToken
    }

    let llm = new ChatOpenAI({
        modelName: "gpt-3.5-turbo-16k",
        streaming: true,
    })

    const fdaPromptTemplate = ChatPromptTemplate.fromMessages([
        ["system", systemMessageFDA],
        ["human", humanMessage]
    ])
    const agent1 = RunnableSequence.from([{
            input: (x: any) => x.input
        },
            fdaPromptTemplate,
            llm,
            new StringOutputParser()
        ]
    )

    let result = await agent1.invoke({
        input: messages[0].content
    })

    result = result.replace("JSON: ", "")


    let parsedResult = JSON.parse(result)
    let searchParams = parsedResult.search_params
    let fieldsToReturn = parsedResult.fields_to_return
    let limit = parsedResult.limit

    const fdaResult = await FDATool(searchParams, fieldsToReturn, limit)

    // const fdaPromptTemplate2 = ChatPromptTemplate.fromMessages([
    //     ["system", systemMessageSummarize],
    //     ["human", apiMessage]
    // ])

    // const agent2 = RunnableSequence.from([{
    //         question: (x: any) => x.question,
    //         information: (x: any) => x.information
    //     },
    //         fdaPromptTemplate2,
    //         llm,
    //         new StringOutputParser()
    //     ]
    // )

    // const stream2 = agent2.stream({
    //     question: messages[0].content,
    //     information: fdaResult
    // })

    // const response = agent2.invoke({
    //     question: messages[(messages.length - 1)].content,
    //     information: fdaResult
    // })

    const res = await openai.createChatCompletion({
        model: 'gpt-3.5-turbo-16k',
        messages,
        temperature: 0.7,
        stream: true
    })

    const stream = OpenAIStream(res, {
        async onCompletion(completion) {
            const title = json.messages[0].content.substring(0, 100)
            const id = json.id ?? nanoid()
            const createdAt = Date.now()
            const path = `/chat/${id}`
            const payload = {
                id,
                title,
                userId,
                createdAt,
                path,
                messages: [
                    ...messages,
                    ["system", systemMessageSummarize],
                    ["human", messages[(messages.length - 1)].content],
                    ["human", fdaResult],
                    {
                        content: completion,
                        role: 'assistant'
                    }
                ]
            }
            await kv.hmset(`chat:${id}`, payload)
            await kv.zadd(`user:chat:${userId}`, {
                score: createdAt,
                member: `chat:${id}`
            })
        }
    })

    return new StreamingTextResponse(stream)
}
