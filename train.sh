# RNN
python char_rnn.py --name rnn --num_epoch 10
python char_rnn.py --name rnn --num_epoch 10 --num_layers 4

# LTM
python char_rnn.py --name ltm --num_epoch 10
python char_rnn.py --name ltm --num_epoch 10 --num_layers 4

# GRU
python char_rnn.py --name gru --num_epoch 10
python char_rnn.py --name gru --num_epoch 10 --num_layers 4

# Pseudo LSTM
# python char_rnn.py --name pseudolstm --num_epoch 10
# python char_rnn.py --name pseudolstm --num_epoch 10 --num_layers 4

# LSTM
python char_rnn.py --name lstm --num_epoch 10
python char_rnn.py --name lstm --num_epoch 10 --num_layers 4

# Read first LSTM
# python char_rnn.py --name readfirstlstm --num_epoch 10
# python char_rnn.py --name readfirstlstm --num_epoch 10 --num_layers 4

# Read first LSTM + matrix memory cell
# python char_rnn.py --name readfirstmmclstm --num_epoch 10
# python char_rnn.py --name readfirstmmclstm --num_epoch 10 --num_layers 4