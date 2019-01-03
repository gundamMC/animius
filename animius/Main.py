import animius.Console as Console

print('Animius. Type help or ? to list commands.')

while True:
    user_input = input('Input: ')

    if user_input.lower() == 'exit':
        break
    elif user_input.lower() == 'help' or '?':
        pass

    command, args = Console.ParseArgs(user_input)

    print(command)
    print(args)

    if command is None:
        continue

    if command in commands:
        commands[command].__call__(args)
    else:
        print('Invalid command')
