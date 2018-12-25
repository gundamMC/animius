import animius as am

from .SocketServerModel import Response


class ArgumentError(Exception):
    pass


class Console:

    def __init__(self):
        self.models = {}
        self.waifu = {}
        self.model_configs = {}
        self.data = {}
        self.embeddings = {}

    @staticmethod
    def check_arguments(args, hard_requirements, soft_requirements):
        # Check if the user-provided arguments meet the requirements of the method/command
        # hard_requirement throws ArgumentError if not fulfilled
        # soft_requirement gives a value of None
        for req in hard_requirements:
            if req not in args:
                raise ArgumentError('{0} is required')

        for req in soft_requirements:
            if req not in args:
                args['req'] = None

    def create_model_config(self, **kwargs):
        Console.check_arguments(kwargs, ['name', 'cls'], ['config', 'hyperparameters', 'data'])

        self.model_configs[kwargs['name']] = am.ModelConfig(kwargs['cls'],
                                                            kwargs['config'],
                                                            kwargs['hyperparameters'],
                                                            kwargs['data'])

    def handle_network(self, request):

        command = request.command.lower().replace(' ', '_')
        method_to_call = getattr(self, command)

        try:
            result = method_to_call(request.arguments)
            if result is None:
                result = {}
            return Response.createResp(request.id, 0, 'success', result)
        except ArgumentError as exc:
            return Response.createResp(request.id, 1, exc, {})
        except Exception as exc:
            return Response.createResp(request.id, 2, exc, {})
