<h1 align="center">FlowerPower</h1>

<div align="center">
  <strong>Classifying flowers using the Flower Dataset</strong>
</div>
<div align="center">
  <code>A tiny project for Complori</code> 
</div>

## Install
* Clone repository
  ```
  git clone https://github.com/sycz00/FlowerPower.git
  ```
* Create Docker image
  ```
  docker build -t flower .
  ```
## Docker Networking and Running
* Assign a static IP to container by defining subnetmasking
```
docker network create --subnet 192.0.2.0/24 flower_power_net
```
* Run Docker image with static ip (remove -d flag if you don't want to run the container in the background)
```
docker run -p 5000:80 -itd --network=flower_power_net --ip=192.0.2.69 flower
```
* Find and Stop Docker service when using -d flag
```
docker ps
```
```
docker stop <CONTAINER ID>
```
Following the instructions until this point, the Docker container should be running

## Using the API
* Inference of example image <code>daisy.png</code> by sending it to static ip assigned in previous step
```
curl -X POST -F "file=@inference_test/daisy.jpg" http://192.0.2.69:5000/predict
```
* Request a new training process, includes overwriting the existing checkpoint with new one. Custom number of epochs and learning-rate can be used here
```
curl -X POST -H "Content-Type: application/json" -d '{"epochs": 10,"lr":0.001}' http://192.0.2.69:5000/train
```



## Process and Decisions
* In this study, a shallow CNN was employed as a feature extractor, consisting of 6 Conv2D layers that incorporate maxpooling for input subsampling and are accompanied by ReLU activation functions. This is then followed by a 3-layered Dense network. The design of the CNN architecture was not driven by specific theoretical considerations, but rather by practical constraints such as time and computational complexity, aimed at enabling local machine execution. An additional experiment was conducted utilizing a pre-trained ResNet-18 model from ImageNet, which unsurprisingly demonstrated superior performance compared to the aforementioned shallow CNN utilized in this study.

* This specific dataset was selected due to its focus on children aged 7 to 16, involving the introduction of a natural relationship centered around flowers. Moreover, I hold the belief that flowers possess inherent beauty, and providing children with the chance to witness practical applications in nature could prove advantageous. Despite the extensive diversity within various datasets available.
  
* Constructing the model proved to be uncomplicated. However, my relatively brief attention to containerizing the model subsequently necessitated a more substantial time investment in that aspect. Initially, I contemplated employing Torchserve, yet I ultimately opted for a combination of Docker and a basic Flask API. While using the Flask app within a Docker container, I encountered an obstacle: an inability to establish local communication with the container via the localhost address. This predicament prompted me to seek out a solution. The sole viable method of communication involved utilizing the dynamically assigned address of the Docker container upon startup. Because this address fluctuated each time, I needed to explore an alternative approach for assigning a static IP. The resolution entailed creating a distinct sub-network and subsequently assigning a fixed address to the Docker container, facilitating communication through the utilization of curl.
  
* This application represents the bare essentials of its potential capabilities. Initial enhancements could involve integrating a graphical interface through HTML, allowing users to choose images via a dropdown menu and subsequently submit them with a design that adapts to different screen sizes. Additionally, the application could incorporate various other machine learning techniques, such as warm-starting or pre-training, or empower users to modify hyperparameters and other critical parameters to fine-tune performance. In essence, providing increased access to the pipeline would give more insights.
* 
* The following image shows the training progress on the validation dataset. it can be seen that the shallow CNN makes reasonable progress.
![Validation curve](val.png)



### Dataset
- [x] [Flower Dataset](https://www.kaggle.com/alxmamaev/flowers-recognition/flowers](https://www.kaggle.com/code/rayankazi/flowers-classification-pytorch)https://www.kaggle.com/code/rayankazi/flowers-classification-pytorch)
