import torch

checkpoint_dir="/project/student_projects2/dhe/BERT/experiments/pytorch_bert_model/pretrained_model_ppl_11.952676_e43.00.pt"
output = "/project/student_projects2/dhe/BERT/experiments/pytorch_bert_model/pretrained_model_ppl_11.pt"
checkpoint = torch.load(checkpoint_dir)
print("before deleting, the keys are:",checkpoint.keys())
for key_item in ['optim', 'dicts', 'opt', 'epoch', 'iteration', 'batch_order', 'additional_batch_order', 'additional_data_iteration', 'amp']:
    if key_item in checkpoint:
        del checkpoint[key_item]

print("after deleting, the keys are:", checkpoint.keys())
save_checkpoint = checkpoint["model"]




torch.save(save_checkpoint, output)