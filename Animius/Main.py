import Animius.Console as Console

print('Welcome to Project Waifu')

while True:
    user_input = input('Input: ')

    if user_input == 'Exit':
        break

    command, args = Console.ParseArgs(user_input)

    print(command)
    print(args)

    if command is None:
        continue

    if command in commands:
        commands[command].__call__(args)
    else:
        print('Invalid command')
