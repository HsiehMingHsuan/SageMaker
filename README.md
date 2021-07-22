

## AWS SageMaker
### 步驟一: 創建 AWS SageMaker Notebook Instance
要使用 AWS SageMaker 有兩種方法，第一個是用 AWS SageMaker Studio，其中的工具比較多。第二個是由自己開 Notebook 的方法。

這裡先進行自己開 Notebook 方法的教學。
首先到 SageMaker 頁面下的 **筆記本**  **筆記本執行個體**
點擊 __建立筆記本執行個體__
就可以對新建立的筆記本取名以及選其使用的硬體 (ml.t2.medium 為最便宜的)
暫不使用 **Elastic Inference**
接下來選擇所需的IAM 角色就可以開
若沒有IAM角色，選擇 **建立新角色** ，並給予 **任何 S3 儲存貯體** 權限
就可以點選 __建立筆記本執行個體__ 
### 步驟二: 創建 jupyter Notebook 
步驟一做完後就能在 **筆記本執行個體** 看到創建的筆記本
筆記本的右方有 **Open Jupyter | Open JupyterLab** 在這裡我們先點擊  **Open Jupyter** 
打開後可以看到jupyter介面
若要創建新的 jupyter notebook (.ipynb) 可以選擇 **new** 並選擇環境創建
也可以點擊 **upload** 從本地端上傳
在這裡我們先點擊 **upload** 從本地端上傳這個 repo 附的 ipynb 檔
### 步驟三: 由 jupyter notebook 端創建 S3 bucket 以存取訓練/測試資料以及模型
打開創建的 jupyter notebook 
可以看到基礎 jupyter notebook 介面
jupyter notebook 中可以選擇要加文字 cell 或是 code cell
對於 code cell 我們能選取後點選 Run (或快捷鍵shift+enter) 一次 run 一個 cell 或是選擇一次直接執行完全部的 code cell
在範例 notebook中，我們先 import 一些要用到的 library
		
	import sagemaker 
	import boto3 
	from sagemaker.amazon.amazon_estimator import get_image_uri
	from sagemaker.session import s3_input, Session
接下來我們就可以開始創建 S3 bucket
		
	bucket_name =  'sagemakertest52657055'  # bucket名稱
	my_region = boto3.session.Session().region_name # 存下此instance的區域
	print(my_region)
	s3 = boto3.resource('s3')
	try:
		if my_region ==  'ap-northeast-1':
			s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={
    'LocationConstraint': my_region})
		print('S3 bucket created successfully')
	except  Exception  as e:
		print('S3 error: ',e)
若成功會有 S3 bucket created successfully 的訊息
此時可以去 AWS S3 上確認是否有創建成功
這裡要注意 bucket 的命名在整個區域要唯一，否則不能建立

	# set an output path where the trained model will be saved
	prefix =  'xgboost-as-a-built-in-algo'
	output_path ='s3://{}/{}/output'.format(bucket_name, prefix)
	print(output_path)
這段主要是在剛剛創建好的 bucket 下指定一個存 model 的路徑
以上這個步驟也可以改成由其他途徑去 S3 create bucket 或是直接 access 舊有的 bucket
### 步驟四:  資料處理以進行訓練
當 S3 建置好後，就可以開始試驗資料如何去訓練
由於我們 S3 Bucket 中還未有任何資料
我們先下載範例資料，這個範例資料簡單來說就是客戶會不會買一個特定產品，其中每一個 row 都是一個客戶，而最後兩個 column 則是會不會買的結果 (target) 

	import pandas as pd
	import urllib
	try:
		urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv",  "bank_clean.csv")
		print('Success: downloaded bank_clean.csv.')
	except  Exception  as e:
		print('Data load error: ',e)
	try:
		model_data = pd.read_csv('./bank_clean.csv',index_col=0)
		print('Success: Data loaded into dataframe.')
	except  Exception  as e:
		print('Data load error: ',e)
這個 cell 跑完就會下載檔案並且把資料讀進來這個 notebook 


	import numpy as np
	train_data, test_data = np.split(model_data.sample(frac=1,  random_state=1729),  [int(0.7  *  len(model_data))])
	print(train_data.shape, test_data.shape)
這裡是把讀進來的資料 73 分成訓練資料 (training set) 跟測試資料 (validation set) ，主因是因為當模型訓練好了要測試結果時，若丟進模型測試的資料是已經被模型訓練過的，想當然正確率會很高/誤差會很低，那會非常容易造成 overfit 的問題，因此預先把資料分成預計要投入訓練的與要來驗證成果的，要注意不一定要 73 分
		
	import os
	pd.concat([train_data['y_yes'], train_data.drop(['y_no',  'y_yes'],
	axis=1)],
	axis=1).to_csv('train.csv',  index=False,  header=False)
	boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix,  'train/train.csv')).upload_file('train.csv')
	s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train'.format(bucket_name, prefix),  content_type='csv')
這裡由於之後要給 xgboost 訓練時需要特別形式 (其他演算法可能不一樣)，就是 target feature 要放在第一個 column 且不需要 header 以及 index
先把訓練資料的 'y_yes' 那一個 column 拿出來再與訓練資料除去 'y_no',  'y_yes' 兩 column 後做合併，之後存入 S3 bucket 中 train 下面

	pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)
	boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
	s3_input_test = sagemaker.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')
這裡是對測試資料做同樣的事
### 步驟五:  載入模型進行訓練
	# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
	# specify the version depending on your preference.
	container = sagemaker.image_uris.retrieve('xgboost', boto3.Session().region_name, version='1.0-1')
載入模型，我們選擇 xgboost
	
	# initialize hyperparameters
	hyperparameters = {
	        "max_depth":"5",
	        "eta":"0.2",
	        "gamma":"4",
	        "min_child_weight":"6",
	        "subsample":"0.7",
	        "objective":"binary:logistic",
	        "num_round":50
	        }
這裡是在設置訓練的超參數，不同演算法所要設置的參數可能不同

	# construct a SageMaker estimator that calls the xgboost-container
	estimator = sagemaker.estimator.Estimator(role=sagemaker.get_execution_role(),
	                                          instance_count=1, 
	                                          instance_type='ml.m5.2xlarge',
	                                          volume_size=5, # 5 GB 
	                                          max_run=300,
	                                          output_path=output_path,
	                                          image_uri=container, 
	                                          use_spot_instances=True,
	                                          max_wait=600,
	                                          hyperparameters=hyperparameters)
這裡在設置 SageMaker 的 Estimator，可以把這個想成自動 train 的套件，詳細可以[參見](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html) 
設置完執行
	
	estimator.fit({'train': s3_input_train,'validation': s3_input_test})
就會開始 train，我這邊試跑的時候是 train 了 49 秒
train 完可以去 S3 bucket 下面 output 確認
### 步驟六:  用 Endpoint 部屬模型
	from sagemaker.predictor import CSVSerializer
	xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge',serializer=CSVSerializer())
這裡執行完就會產生 Endpoint 
可以執行
	
	xgb_predictor.endpoint_name
來看 Endpoint 名稱，其形式為 **"sagemaker-xgboost-YYYY-MM-DD-HH-MM-SS-SSS"**
建置完可去 **SageMaker**  **推論**  **端點** 查看
在這裡如果 predictor 想要用之前就部屬過的 Endpoint ，可以參考[這裡](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-model-deployment.html)

	xgb_predictor_reuse=sagemaker.predictor.Predictor( endpoint_name="`sagemaker-xgboost-YYYY-MM-DD-HH-MM-SS-SSS`", 
		sagemaker_session=sagemaker.Session(), 
		serializer=sagemaker.serializers.CSVSerializer() )
### 步驟七:  模型評估
	test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
	predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
	predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
	print(predictions_array.shape)
這裡執行完就會把測試資料輸入模型進行預測，並存下預測結果
接下來可以簡單評估預測結果

	cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
	tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
	print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
	print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
	print("Observed")
	print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
	print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))
這邊執行完會產出一個準確率與一個簡單的 confusion matrix 
### 步驟八:  刪除 Endpoint 、資料與環境
試驗結束後要刪除 Endpoint 、資料與環境
執行下面就能了

	sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
	bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
	bucket_to_delete.objects.all().delete()
結束後去 **SageMaker**  **推論**  **端點** 確認是否刪除
再去 **SageMaker**  **推論**  **端點組態**  手動刪除端點組態、**SageMaker**  **推論**  **模型** 手動刪除模型
最後去 S3 bucket 裡面確認建置的 bucket 是否為空，若為空的就能把 bucket 刪除
接下來關掉 jupyter notebook 就能在 jupyter 首頁這邊勾選全部檔案 (.ipynb & .csv) 進行刪除 
刪除後就能關閉 jupyter 頁面
並在 SageMaker **筆記本**  **筆記本執行個體** 下選擇之前建立的 notebook instance， 在 **動作** 選 **停止**，等它停止完後就可以 **動作** 選 **刪除**
在 SageMaker 的儀表板中可以確認資源情況，要注意儀表板中的 **訓練任務** 以及 **處理任務** 可以想成是資料訓練跟資料處理的紀錄，是無法刪除的
> Written with [StackEdit](https://stackedit.io/).
