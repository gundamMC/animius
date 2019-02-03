import animius as am
import readline


console = am.Console()
console.init_commands()


def completer(user_input, state):
    options = [i for i in console.commands if i.startswith(user_input)]
    if state < len(options):
        return options[state]
    else:
        return None


readline.parse_and_bind("tab: complete")
readline.set_completer(completer)

print("Animius. Type 'help' or '?' to list commands.")

while True:
    user_input = input('Input: ')

    if user_input.lower() == 'exit':
        break

    console.handle_command(user_input)
