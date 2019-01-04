import animius as am
import readline


def completer(user_input, state):
    options = [i for i in commands if i.startswith(user_input)]
    if state < len(options):
        return options[state]
    else:
        return None


readline.parse_and_bind("tab: complete")
readline.set_completer(completer)

console = am.Console()

commands = {
    '--save': console.save,
    '-s': console.save,
    '--create_model_config': console.create_model_config()
}
print("Animius. Type 'help' or '?' to list commands.")

while True:
    user_input = input('Input: ')

    if user_input.lower() == 'exit':
        break
    elif user_input.lower() == 'help' or '?':
        pass
    elif user_input.lower() == 'about' or 'version':
        pass

    command, args = am.Console.ParseArgs(user_input)

    print(command)
    print(args)

    if command is None:
        continue

    if command in commands:
        commands[command].__call__(args)
    else:
        print('Invalid command')
