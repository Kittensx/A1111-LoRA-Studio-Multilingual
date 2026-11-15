# Languages, help and documentation discovery

Switch the interface language and find related guides.

The language picker in LoRA Studio controls UI labels and this Help viewer. Your prompts, model names and custom preset text do not change language automatically.

## Documentation registry
- Click a **?** icon beside a Studio control to open a relevant topic.
- Search for topics in the Help viewer and follow **Related topics** links.
- The registry discovers `help_documentation/<locale>/<topic>.md` files at runtime.
- Optional `help_documentation/registry/<topic>.json` files configure related topics and context links.
- When a help article is unavailable in your selected language, the viewer shows the English version and labels the fallback.

A new article or translation becomes available when the folder is scanned again (reopening Help or using Reload); no JavaScript changes are needed. The A1111 web server must be restarted after installing new Python runtime code.
