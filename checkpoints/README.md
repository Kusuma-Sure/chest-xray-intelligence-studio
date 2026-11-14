Copy matching trained checkpoint files here:
- New pipeline: vit.pt, densenet.pt, capsnet.pt, gat.pt
- Earlier DenseNet notebook: best_densenet121_chexpert.pth
- Earlier CapsNet notebook: best_capsnet_10epoch.pth or best_capsnet_weighted.pth

The API strictly checks checkpoint key/shape compatibility at first load. The old Hugging Face ViT folder is not compatible with this TorchVision ViT implementation. GAT requires training with this project. Also copy generated *_summary.json and *_metrics.csv files to populate real benchmark rows. Do not use document-forgery checkpoints.
