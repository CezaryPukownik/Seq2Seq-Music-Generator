## MUSIC GENERATION USING DEEP LEARNING
## AUTHOR: CEZARY PUKOWNIK

## How to use:
1. In folder ./midi_packs make folder with midi files you want train on
2. Use extract.py to export data from midis 
  > ./extract.py [str: midi_pack_name] [str: name_of_session] --b [int: seq_len] -a [analize data first]
3. Use train.py to train model
  > ./train.py [str: name_of_session] --b [int: batch_size] --l [int: latent_space] --e [int: epochs] --i [str: instrument] -r [reset]
4. Use generate.py to generate music from models
  > ./generate.py [str: name_of_session] --n [number of generations] --m [mode {'from_seq','from_state'}]

