import faulthandler
import fileinput
import json
import signal

from ttad.ttad_model.ttad_model_wrapper import ActionDictBuilder

faulthandler.register(signal.SIGUSR1)

if __name__ == "__main__":
    # ttad_model_path = os.path.join(os.path.dirname(__file__), "models/ttad/ttad.pth")
    # ttad_embedding_path = os.path.join(os.path.dirname(__file__), "models/ttad/ttad_ft_embeds.pth")
    # ttad_model = ActionDictBuilder(ttad_model_path, ttad_embedding_path)
    ttad_model = ActionDictBuilder()

    print("> ", end="", flush=True)
    for line in fileinput.input():
        action_dict = ttad_model.parse(["human: " + line.strip()])
        print(json.dumps(action_dict))
        print("\n> ", end="", flush=True)
