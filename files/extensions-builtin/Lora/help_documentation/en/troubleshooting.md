# Troubleshooting LoRA Studio

Common problems and safe recovery steps.

If a LoRA is missing, refresh the LoRA list and confirm its file is located in an indexed LoRA directory.

If a preset or tag cannot be applied, check the exact prompt syntax, model profile, and any validation message. Avoid forcing an unsupported architecture or block layout.

If the Help viewer reports unavailable documentation, confirm the `extensions-builtin/Lora/help_documentation` folder was installed and that the A1111 application was restarted after deploying the help API.

If the package changes a core A1111 file unexpectedly, use the installer backup and rollback commands. Test on a backed-up A1111 installation before production use.
