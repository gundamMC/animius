from ProjectWaifu.SpeakerVerification.SpeakerVerificationNetwork import SpeakerVerificationNetwork
from ProjectWaifu.IntentNER.IntentNERNetwork import IntentNERNetwork
from shlex import split

Networks = {}
SelectedNetwork = None


def ParseArgs(InputString):
    return split(InputString)


def GetNetworks(args):
    print("Current networks:")
    for name, value in Networks.items():
        print(name, '-', type(value).__name__)


def Select(args):
    if len(args) != 1:
        print("Usage: Select <Network Name>")
        return
    if args[0] not in Networks:
        print("Error: Network \"" + args[0] + "\" does not exist")
        return
    global SelectedNetwork
    SelectedNetwork = Networks[args[0]]


def GetSelected(args):
    return SelectedNetwork.name


def AddNetwork(args):

    if len(args) != 2:
        print("Usage: AddNetwork <Network Type> <Network Name>")
        return

    if args[0].lower() == "speakerverification":
        Networks[args[1]] = SpeakerVerificationNetwork()
        print("Speaker verification network \"" + args[1] + "\" added and selected!")

    elif args[0].lower() == "intentandner":
        Networks[args[1]] = IntentNERNetwork()
        print("Intent and NER network \"" + args[1] + "\" added and selected!")

    else:
        print("Invalid type")
        return

    Select([args[1]])


def SetTrainingData(args):

    if len(args) < 1:
        print("Usage: SetTrainingData {<Network Arguments>}")
        return

    if isinstance(SelectedNetwork, SpeakerVerificationNetwork):

        if len(args) != 2:
            print("Usage: SetTrainingData <True Paths> <False Paths>")
            return

        TruePaths = args[0]
        FalsePaths = args[1]
        SelectedNetwork.setTrainingData(TruePaths, FalsePaths)
        print("Training data set!")


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


def Predict(args):
    if len(args) != 1:
        print("Usage: Predict <Network input>")
        return
    output = SelectedNetwork.predict(args[0])
    print(output)


def PredictAll(args):
    if len(args) == 1:
        output = SelectedNetwork.predictAll(args[0])
    elif len(args) == 2:
        output = SelectedNetwork.predictAll(args[0], args[1])
    else:
        print("Usage: Predict <Path of file containing network input>")
        return
    print(output)
