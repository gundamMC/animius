import ProjectWaifu.SpeakerVerification as SpeakerVerification
import ProjectWaifu.IntentNER as IntentNER
import ProjectWaifu.Chatbot as Chatbot
import ProjectWaifu.ModelClasses as ModelClasses
import ProjectWaifu.Utils as Utils
from shlex import split as shell_split


class Console:

    Networks = {}
    SelectedNetwork = None

    @staticmethod
    def GetNetworks(args):
        print("Current networks:")
        for name, value in Networks.items():
            print(name + ' - ' + type(value).__name__)

    @staticmethod
    def Select(args):
        if len(args) != 1:
            print("Usage: Select <Network Name>")
            return
        if args[0] not in Networks:
            print("Error: Network \"" + args[0] + "\" does not exist")
            return
        global SelectedNetwork
        SelectedNetwork = Networks[args[0]]

    @staticmethod
    def GetSelected(args):
        return SelectedNetwork.name

    @staticmethod
    def AddNetwork(args):

        if len(args) != 2:
            print("Usage: AddNetwork <Network Type> <Network Name>")
            return

        if args[0].lower() == "speakerverification":
            Networks[args[1]] = SpeakerVerificationModel()
            print("Speaker verification network \"" + args[1] + "\" added and selected!")

        elif args[0].lower() == "intentandner":
            Networks[args[1]] = IntentNERModel()
            print("Intent and NER network \"" + args[1] + "\" added and selected!")

        else:
            print("Invalid type")
            return

        Select([args[1]])

    @staticmethod
    def SetTrainingData(args):

        if len(args) < 1:
            print("Usage: SetTrainingData {<Network Arguments>}")
            return

        SelectedNetwork.setTrainingData(args)
        print("Training data set!")

    @staticmethod
    def Train(args):

        if len(args) > 2:
            print("Usage: Train [Epochs] [Display step]")
            return

        if len(args) == 0:
            SelectedNetwork.train()
        elif len(args) == 1:
            SelectedNetwork.train(epochs=int(args[0]))
        else:
            SelectedNetwork.train(epochs=int(args[0]), display_step=int(args[1]))

    @staticmethod
    def Predict(args):
        if len(args) != 1:
            print("Usage: Predict <Network input>")
            return
        output = SelectedNetwork.predict(args[0])
        print(output)

    @staticmethod
    def PredictAll(args):
        if len(args) == 1:
            output = SelectedNetwork.predictAll(args[0])
        elif len(args) == 2:
            output = SelectedNetwork.predictAll(args[0], args[1])
        else:
            print("Usage: Predict <Path of file containing network input>")
            return
        print(output)

    @staticmethod
    def LoadWordVector(args):
        if len(args) == 1:
            Utils.createEmbedding(args[0])
        else:
            print("Usage: LoadWordVector <Path of word vector>")
            return
            print(output)

    commands = {
        's': AddNetwork
    }

    @staticmethod
    def ParseArgs(input_string):
        try:
            input_string = shell_split(input_string)
        except Exception as e:
            print(e)
            return None, None
        command = input_string[0]
        args = {}
        for arg in input_string[1:]:
            if not str.startswith(arg, '-') or '=' not in arg:
                print('Invalid argument', arg, '. Arguments must start with - or -- and include a =')
                return None, None
            arg_name, arg_value = arg.lstrip('-').split('=')
            args[arg_name] = arg_value

        return command, args
