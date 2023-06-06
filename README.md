# URLComSum

This code repository is for paper [Efficient and Interpretable Compressive Text Summarisation with Unsupervised Dual-Agent Reinforcement Learning], Workshop on Simple and Efficient Natural Language Processing (SustaiNLP), co-located with ACL 2023

### To train the model:
> python train_OT_hybridcompressivepointer.py --experiment newsroom_2_26 --dataset_str newsroom --dataset_doc_field text --max_ext_output_length 2 --max_comp_output_length 26 --train_batch_size 3 --tkner w2v

### To test the model:
> python decode_full_model_hybridcompressorpointer.py --dataset_str newsroom --dataset_doc_field text --path=decoded_newsroom_2_26 --test --model_dir=models --model_name=summarizer_newsroom_2_26_ckpt.bin  --max_ext_output_length 2 --max_comp_output_length 26 --tkner w2v
