# Compile model with Neo

如果要 model 能夠平順且能夠在 local device 上達到最好的效果，我們必須要在 local device 上使用 [SageMaker Neo](https://docs.aws.amazon.com/sagemaker/latest/dg/neo.html) compile 過的 model

*prerequisites: model trained and is on S3 bucket*

在 AWS 上要使用 Neo compile 有三種作法，分別為 

* CLI

* Console

* SDK

## Compile Neo with CLI

這裡的做法主要是參考[這邊](https://docs.aws.amazon.com/sagemaker/latest/dg/neo-job-compilation-cli.html)

**步驟一**: 創建一個符合 [CreateCompilationJob](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateCompilationJob.html) API 的 json file, 格式請參考 [這邊](https://docs.aws.amazon.com/sagemaker/latest/dg/neo-job-compilation-cli.html) 的 Example

其中會要指定 deploy 的硬軟體環境, 請依照裝置需求選取, 像我是要在 Raspberry Pi 4 上面跑, 就會去照 [Device Exmaple](https://docs.aws.amazon.com/sagemaker/latest/dg/neo-job-compilation-cli.html) 去 configure 然後在 "TargetDevice" 指定 "rasp4", "RoleArn" 要選一個有要存 compile 成品的 S3 bucket 權限的 IAM role

Neo 支援的裝置與系統詳細內容[在此](https://docs.aws.amazon.com/sagemaker/latest/dg/neo-supported-devices-edge-devices.html)

這邊要特別注意 "DataInputConfig" 跟當初 train model 的 ML framework 有關, 請參照[這裡](https://docs.aws.amazon.com/sagemaker/latest/dg/neo-compilation-preparing-model.html)設置

**步驟二**: AWS CLI 登入後在 terminal 執行

    aws sagemaker create-compilation-job --cli-input-json file://job.json --region ap-northeast-1 
    
這邊 json 指定剛剛創建的 json, 區域選自己要的區域

這邊執行完就會開始 Neo compile

可以用

    aws sagemaker describe-compilation-job --compilation-job-name $JOB_NM --region ap-northeast-1

來監控 compile 狀況, 這邊 $JOB_NM 改成之前指定的 CompilationJobName

也可以用

    aws sagemaker stop-compilation-job --compilation-job-name $JOB_NM --region ap-northeast-1

停止 compile

想要看全部 compile job 可以用

    aws sagemaker list-compilation-jobs --region ap-northeast-1

來看

若要上 console 觀看 compile 情形, 可以去 **SageMaker** **推論** **編譯任務** 上面看

## Compile Neo with Amazon SageMaker Console

這部分比較直觀，詳細請[參考](https://docs.aws.amazon.com/sagemaker/latest/dg/neo-job-compilation-console.html)

## Compile using Amazon SageMaker Python SDK

這部分主要是用 SageMaker SDK 中的 compile_model method

請參考[這裡](https://docs.aws.amazon.com/sagemaker/latest/dg/neo-getting-started-edge-step1.html)