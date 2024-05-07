python prompt_generation.py \
--model_name_or_path /home/test/cj/DSD-RE/lm_model/t5-large \
--model t5 \
--lr 5e-5 \
--device 0 \
--epoch 50 \
--save_path ./checkpoints/t5-large-encoder-only.ckpt \
--batch_size 8 \
--using_decoder_past_key_values False \
--mode eval