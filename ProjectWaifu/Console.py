from ProjectWaifu.SpeakerVerification.SpeakerVerificationNetwork import SpeakerVerificationNetwork
from ProjectWaifu.IntentNER.IntentNERNetwork import IntentNERNetwork
from shlex import split
import ProjectWaifu.Utils as Utils

Networks = {}
SelectedNetwork = None


def ParseArgs(InputString):
    return split(InputString)


def GetNetworks(args):
    Utils.printMessage("Current networks:")
    for name, value in Networks.items():
        Utils.printMessage(name, '-', type(value).__name__)


def Select(args):
    if len(args) != 1:
        Utils.printMessage("Usage: Select <Network Name>")
        return
    if args[0] not in Networks:
        Utils.printMessage("Error: Network \"" + args[0] + "\" does not exist")
        return
    global SelectedNetwork
    SelectedNetwork = Networks[args[0]]


def GetSelected(args):
    return SelectedNetwork.name


def AddNetwork(args):

    if len(args) != 2:
        Utils.printMessage("Usage: AddNetwork <Network Type> <Network Name>")
        return

    if args[0].lower() == "speakerverification":
        Networks[args[1]] = SpeakerVerificationNetwork()
        Utils.printMessage("Speaker verification network \"" + args[1] + "\" added and selected!")

    elif args[0].lower() == "intentandner":
        Networks[args[1]] = IntentNERNetwork()
        Utils.printMessage("Intent and NER network \"" + args[1] + "\" added and selected!")

    else:
        Utils.printMessage("Invalid type")
        return

    Select([args[1]])


def SetTrainingData(args):

    if len(args) < 1:
        Utils.printMessage("Usage: SetTrainingData {<Network Arguments>}")
        return

    if isinstance(SelectedNetwork, SpeakerVerificationNetwork):

        if len(args) != 2:
            Utils.printMessage("Usage: SetTrainingData <True Paths> <False Paths>")
            return

        TruePaths = args[0]
        FalsePaths = args[1]
        SelectedNetwork.setTrainingData(TruePaths, FalsePaths)
        Utils.printMessage("Training data set!")


def Train(args):

    if len(args) > 2:
        Utils.printMessage("Usage: Train [Epochs] [Display step]")
        return

    if len(args) == 0:
        SelectedNetwork.train()
    elif len(args) == 1:
        SelectedNetwork.train(epochs=int(args[0]))
    else:
        SelectedNetwork.train(epochs=int(args[0]), display_step=int(args[1]))


def Predict(args):
    if len(args) != 1:
        Utils.printMessage("Usage: Predict <Network input>")
        return
    output = SelectedNetwork.predict(args[0])
    Utils.printMessage(output)


def PredictAll(args):
    if len(args) == 1:
        output = SelectedNetwork.predictAll(args[0])
    elif len(args) == 2:
        output = SelectedNetwork.predictAll(args[0], args[1])
    else:
        Utils.printMessage("Usage: Predict <Path of file containing network input>")
        return
        Utils.printMessage(output)
