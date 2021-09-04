# import torch
# import model
# from src.submission.run import mconf
#
# model = model.GPT(mconf)
#
# model_file = "vanilla.pretrain.params"
#
# model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
#
# torch.save(model.state_dict(), model_file + "_cpu")
