from shlex import split
import ProjectWaifu.Console as Console

print("Welcome to Project Waifu")


def ParseArgs(InputString):
    return split(InputString)


while True:
    UserInput = input("Input: ")

    if UserInput == "Exit":
        break

    InputArgs = ParseArgs(UserInput)
    Command = InputArgs[0]
    InputArgs = InputArgs[1:]

    method_to_call = getattr(Console, Command, None)

    if method_to_call is None:
        print("Invalid command")
    else:
        method_to_call(InputArgs)
