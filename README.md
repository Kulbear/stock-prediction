# Stock Prediction with Recurrent Neural Network

Stock price prediction with RNN. The data we used is from the Chinese stock.

## Requirements

- Python 3.5
- TuShare 0.7.4
- Pandas 0.19.2
- Keras 1.2.2
- Numpy 1.12.0
- scikit-learn 0.18.1
- TensorFlow 1.0 (GPU version recommended)

I personally recommend you to use Anaconda to build your virtual environment. And the program probably cost a significant time if you are not using the GPU version Tensorflow.

## Get Data

You can run `fetch_data.py` to get a piece of test data. Without changing the script, you can get two seperated csv file named:

- `000002-from-1995-01-01.csv` =====> Contains general data for stock 000002 from 1995-01-01 to today.
- `000002-3-year.csv` =====> Contains candlestick chart data for stock 000002 (万科A) for the most recent 3 years.

You are expected to see results look like (the first DataFrame contains general data where the the second contains detailed candlestick chart data):

```
$ python3 fetch_data.py
[Getting data:]#########################################################################################
Saving DataFrame:
     open   high    low      volume        amount  close
0  20.64  20.64  20.37  16362363.0  3.350027e+08  20.56
1  20.92  20.92  20.60  21850597.0  4.520071e+08  20.64
2  21.00  21.15  20.72  26910139.0  5.628396e+08  20.94
3  20.70  21.57  20.70  64585536.0  1.363421e+09  21.02
4  20.60  20.70  20.20  45886018.0  9.382043e+08  20.70

Saving DataFrame:
     open   high    low     volume  price_change  p_change     ma5    ma10  \
0  20.64  20.64  20.37  163623.62         -0.08     -0.39  20.772  20.721
1  20.92  20.92  20.60  218505.95         -0.30     -1.43  20.780  20.718
2  21.00  21.15  20.72  269101.41         -0.08     -0.38  20.812  20.755
3  20.70  21.57  20.70  645855.38          0.32      1.55  20.782  20.788
4  20.60  20.70  20.20  458860.16          0.10      0.48  20.694  20.806

     ma20      v_ma5     v_ma10     v_ma20  close
0  20.954  351189.30  388345.91  394078.37  20.56
1  20.990  373384.46  403747.59  411728.38  20.64
2  21.022  392464.55  405000.55  426124.42  20.94
3  21.054  445386.85  403945.59  473166.37  21.02
4  21.038  486615.13  378825.52  461835.35  20.70
```

## Demo

<div style="text-align:center">
	<img src="https://cloud.githubusercontent.com/assets/14886380/25383467/de39614e-29ee-11e7-9a3c-ac9e34720b54.png" alt="Training Result Demo" style="width: 450px;"/>
</div>

## Reference

- [Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
- [Understanding LSTM Networks by Christopher Olah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
