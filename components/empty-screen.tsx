import { UseChatHelpers } from 'ai/react'

import { Button } from '@/components/ui/button'
import { ExternalLink } from '@/components/external-link'
import { IconArrowRight } from '@/components/ui/icons'

const exampleMessages = [
  {
    heading: 'What are side-effects of Xarelto?',
    message: `What are side-effects of Xarelto?`
  },
  {
    heading: 'Are there any pregnancy warnings for lisinopril?',
    message: 'Are there any pregnancy warnings for lisinopril?'
  },
  {
    heading: 'What is the mechanism of action of Opdivo?',
    message: `What is the mechanism of action of Opdivo?`
  }
]

export function EmptyScreen({ setInput }: Pick<UseChatHelpers, 'setInput'>) {
  return (
    <div className="mx-auto max-w-2xl px-4">
      <div className="rounded-lg border bg-background p-8">
        <h1 className="mb-2 text-lg font-semibold">
          Welcome to FDA GPT.
        </h1>
        <p className="mb-2 leading-normal text-muted-foreground">
          This is an AI-powered chatbot that answers questions based on data from the {' '}
          <ExternalLink href="https://www.fda.gov/">United States Food and Drug Administration (FDA)</ExternalLink>
          .
        </p>
        <p className="leading-normal text-muted-foreground">
          You can start a conversation here or try the following examples:
        </p>
        <div className="mt-4 flex flex-col items-start space-y-2">
          {exampleMessages.map((message, index) => (
            <Button
              key={index}
              variant="link"
              className="h-auto p-0 text-base"
              onClick={() => setInput(message.message)}
            >
              <IconArrowRight className="mr-2 text-muted-foreground" />
              {message.heading}
            </Button>
          ))}
        </div>
      </div>
    </div>
  )
}
