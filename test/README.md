## TunePref Test Folder

This testfolder contains files and sample code in its early stages, that gives a basic gist of what this whole thing is about. This is, in no way, any indication of the final version of the code, but rather the *starting point* of it.

There are a few files present in the test folder that be observed:
- `testfile_twofeatures.py`
- `music1.csv`
- `frozen-requirements.txt`
- `FLAGS.txt`

The first file, `testfile_twofeatures.py` is a Python code which gives a small glimpse into the workings of Machine Learning. It takes into account two numerical features, and one numerical label to train a predictive model based on said values.

The next file is `music1.csv`, which is a **CORGIS** Dataset that is currently being used for our project. It contains 35 columns of values that spreads over 10000 records.

At the time of testing, the system that was used had **Python 3.5.4 x64** installed (Windows), with the various dependencies listed in the `frozen-requirements.txt`.

`FLAGS.txt` is a text file that indicates the four different flags can be used to suppress different outputs, should the need be. These flags are entered during executing the Python code, and are passed as command line arguments.

> In this project, Machine Learning makes use of **TensorFlow, Pandas, and NumPy** heavily. The records are read from the comma-delimited CSV file, and stored into a Pandas DataFrame. NumPy are used for the intermediate calculations.

##### A small instance of the output window during the execution of the code is:

```python
>> python testfile_twofeatures.py


      artist.hotttnesss           artist.id  ...                      title  year
9098              0.393  ARCPY7O1187FB5C0B7  ...             Stop The World  2004
9313              0.541  AR2O3B51187B98CD94  ...          Come On Back Home  1998
5902              0.357  ARAG1DX1187B98BB10  ...        Blowin' In the Wind  2009
4009              0.464  AR4BDNG1187FB44870  ...          Looking For Clues  1980
2466              0.767  ARXPPEY1187FB51DF4  ...   Just A Little Bit Of You  1975
...                 ...                 ...  ...                        ...   ...
4545              0.482  ARD8JVH1187FB4DA04  ...                 Simple Man  1976
8942              0.399  ARAAP251187B99DDE0  ...       Venite Pa' Maracaibo  2000
9810              0.373  AR0WAW61187B98B7D7  ...              Angel In Hell     0
8033              0.843  ARF5M7Q1187FB501E8  ...           Through With You  2002
6082              0.403  ARUEYGC1187FB3D94B  ...                       Toby  1974
```

##### TensorFlow (v1.8) was installed using:

```python
>> pip install --upgrade tensorflow
```

and as such performs calculations using the CPU, and not GPU (which can greatly speed up the processing speed, depending on what hardware is present in the system).

**_Stay tuned for further updates!_**
