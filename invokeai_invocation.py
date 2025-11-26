import json
from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    invocation,
    invocation_output,
    InputField,
    OutputField,
    UIComponent
)

# Import the separated logic
from parsifal import GrammarParser, StopParsingException

@invocation_output("parsifal_output")
class ParsifalOutput(BaseInvocationOutput):
    """Output for the Parsifal Prompt Builder node, including variables."""
    prompt: str = OutputField(description="The generated prompt")
    variables: str = OutputField(description="JSON string containing all variables set during generation")


@invocation("parsifal", title="Parsifal Prompt Builder", tags=["prompt", "grammar", "parser", "text"], category="prompt", version="1.0.0")
class ParsifalInvocation(BaseInvocation):
    """Generates a prompt using a custom grammar with files, logic and randomness."""

    template: str = InputField(default="", description="The prompt template", ui_component=UIComponent.Textarea)
    root_directory: str = InputField(default="", description="Root path for [file] commands")
    seed: int = InputField(default=0, description="Seed for randomness")

    def invoke(self, context: InvocationContext) -> ParsifalOutput:
        parser = GrammarParser(self.root_directory, self.seed)
        try:
            output_text = parser.parse(self.template)
        except StopParsingException:
            output_text = "" 
        
        vars_json = json.dumps(parser.vars, indent=2)

        return ParsifalOutput(prompt=output_text, variables=vars_json)