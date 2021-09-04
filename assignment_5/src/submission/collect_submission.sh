rm -f assignment5_submission.zip
zip -r assignment5_submission.zip attention.py run.py model.py dataset.py utils.py trainer.py\
    birth_dev.tsv birth_places_train.tsv wiki.txt\
    vanilla.model.params vanilla.finetune.params synthesizer.finetune.params vanilla.nopretrain.dev.predictions vanilla.nopretrain.test.predictions vanilla.pretrain.dev.predictions vanilla.pretrain.test.predictions synthesizer.pretrain.dev.predictions synthesizer.pretrain.test.predictions
