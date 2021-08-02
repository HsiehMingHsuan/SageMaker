# Model inference on local device

建議直接看 [ipynb 檔](./test_deploy.ipynb)的筆記，內容跟這邊一樣但是有 output 能看

*prerequisites: model trained, compiled with neo, and stored on S3 && AWS CLI logged in*

這裡是示範如何在本地端透過從 S3 下載用 **Neo** compile 過的模型並用此模型做出 inference

下面我們會一步一步的講解

我測試使用的環境由於當初 Neo compile 時指定的裝置是 Raspberry Pi 4 B，所以這邊我是用 Rpi 4B, ubuntu 的環境

    !pip install boto3

>

    import boto3

    session = boto3.Session(profile_name='Ming') 
    s3 = session.resource('s3')

這裡簡單測試在 python 下面跑 AWS services, 成功的話所有 S3 bucket 會 print 出來

    # Print out bucket names
    for bucket in s3.buckets.all():
        print(bucket.name)

下載 neo-compile 過的 model

bucket 指定該 model 存放的 bucket, object_path 是 neo-compile 過的 model 在 bucket 下面的路徑， 最後一個變數是下載下來的名稱

    s3client = session.client('s3')
    # Download compiled model locally to edge device
    bucket='sagemaker-neo-compile-test-20210723' # specify the bucket where the neo-compiled model stored
    object_path = 'xgboost-as-a-built-in-algo/output/sagemaker-xgboost-2021-07-23-neo-compiled-rasp4b/model-rasp4b.tar.gz' # specify the path of the model in the bucket
    neo_compiled_model = 'compiled-rasp4b.tar.gz' # download as this name 
    s3client.download_file(bucket, object_path, neo_compiled_model)

建立資料夾放 model, 解壓縮在裡面

    !mkdir ./dlr_model # make a directory to store your model (optional)
    !tar -xzvf ./compiled-rasp4b.tar.gz --directory ./dlr_model 

下面的 cell 執行前，要先在本地安裝 dlr, 如果是要在 x86_64 的 CPU 上面跑，可以直接執行

    !pip install dlr

如果要用 GPU 跑或是非 x86 的 ISA 上， 請[參閱](https://neo-ai-dlr.readthedocs.io/en/latest/install.html#)

Raspberry Pi 4 B 的話可以直接執行下面

    !pip install https://neo-ai-dlr-release.s3-us-west-2.amazonaws.com/v1.9.0/rasp4b/dlr-1.9.0-py3-none-any.whl

>
    import dlr

    device = 'cpu'
    model = dlr.DLRModel('./dlr_model', device)

有了 dlr (Deep Learning Runtime) 我們才能在本地跑 Neo-compiled model

路徑指定之前解壓縮 model 的路徑

這步結束後其實就能用

    out = model.run(x)

的 code 跑 inference 了，其中 out 為模型輸出結果， x 是輸入的資料

要注意當初 Neo-compile 時有指定 Data input configuration

所以這邊使用時也要注意輸入資料的 shape

這個範例當初 Data input configuration 是 {"data":[1, 59]}， 因此輸入的資料形狀要為二維且建議 shape 符合 (num, 59), 這邊 num 為要做 inference 的 data 數量 59 為輸入 feature 數量

輸入 data 的 feature 數量若小於訓練時給的數量 (在這裡為59) 經測試其實 model 還是能做 inference， 但是不建議因為正確率會下降
 
下面就來測試在本地跑一樣的資料是否跟在 SageMaker 上產生的結果一樣

來測試模型是否有正確作用

這裡先下載並讀取之前在 SageMaker 上測試的資料

    import pandas as pd
    import urllib
    try:
        urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
        print('Success: downloaded bank_clean.csv.')
    except Exception as e:
        print('Data load error: ',e)

    try:
        model_data = pd.read_csv('./bank_clean.csv',index_col=0)
        print('Success: Data loaded into dataframe.')
    except Exception as e:
        print('Data load error: ',e)

照著之前的方法切出測試資料

    ### Train Test split

    import numpy as np
    train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
    print(train_data.shape, test_data.shape)
    把測試資料除了 feature 的 column 移除
    test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array

看一下測試資料的 shape

    test_data_array.shape

這邊我們先測試只用一個 data 去做 inference

取測試資料中的第一個

    single_inference_data = test_data_array[0]

確認 shape

這個 shape 不能輸入模型

    single_inference_data.shape

用 NumPy 中的 expand_dims 來變換 shape

這部分也可以用其他方法， 像是 numpy.reshape

    single_inference_data = np.expand_dims(single_inference_data, axis=0)

再確認一下 shape

    single_inference_data.shape

丟入模型 inference, 就有結果了

    single_predictions = model.run(single_inference_data) # predict!

這邊把全部測試資料丟入模型

    predictions = model.run(test_data_array)# predict!

這邊 predictions 出來是 list, 把它轉成 np.array

    type(predictions)
    predictions = np.array(predictions)

這邊轉完物件看一下 shape

    predictions.shape

這邊用 Numpy 裡面的 squeeze 降維以利後面進行資料處理

    predictions = predictions.squeeze()
    predictions.shape

資料分析的部份 code 跟之前在 SageMaker 上跑得一樣

可以看到輸出結果跟之前在 SageMaker Notebook instance 上的一樣

所以 model 有成功在本地端作用
