from flask import Flask, request, jsonify,render_template
import os
from model import FlowerPowerNet
from utils import get_default_device, DeviceDataLoader, to_device, fit, evaluate, plot_accuracies
import torchvision
import torch
from PIL import Image
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader

MAPPING_CLASS_TO_FLOWER = {0:"daisy",1:"dandelion",2:"rose",3:"sunflower",4:"tulip"}

model = FlowerPowerNet(path=f"{os.path.dirname(os.path.realpath(__file__))}/last_chkp.pth")

DEVICE = get_default_device()

transforms_train = torchvision.transforms.Compose(
    [  # Applying Augmentation
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
    )  
transforms_inference = torchvision.transforms.Compose(
    [ 
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
    )  

def flask_service():
    app = Flask(__name__)
    
    @app.route('/predict', methods=['POST'])
    @torch.no_grad()
    def predict():
        model.load_state_dict(DEVICE)
        model.to(DEVICE)
        model.eval()
        img = Image.open(request.files['file'])
        img = transforms_inference(img).unsqueeze(0).to(DEVICE)
        _,prediction = torch.max(model(img).detach(), dim=1)#.item()

        return jsonify({'prediction': MAPPING_CLASS_TO_FLOWER[prediction.item()]}), 200

    @app.route('/train', methods=['POST'])
    def train():
        
        data = request.get_json()
        if ('epochs' not in data or 'lr' not in data):
            return jsonify({'Error': 'Missing parameters'}), 400

        base_dir = f"{os.path.dirname(os.path.realpath(__file__))}/flowers"
        dataset = ImageFolder(base_dir, transform=transforms_train)

        validation_size = 400
        training_size = len(dataset) - validation_size
        dataset.classes
        train_ds, val_ds_main = random_split(dataset,[training_size, validation_size])
        val_ds, test_ds  = random_split(val_ds_main,[200, 200]) 
        
        train_dl = DataLoader(train_ds, batch_size = 32, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size = 32)
        test_dl = DataLoader(test_ds, batch_size = 32)

        
        train_dl = DeviceDataLoader(train_dl, DEVICE)
        val_dl = DeviceDataLoader(val_dl, DEVICE)
        model = FlowerPowerNet(path=f"{os.path.dirname(os.path.realpath(__file__))}/last_chkp.pth")
        model_train = to_device(model, DEVICE)
        
        num_epochs = data['epochs']

        opt_func = torch.optim.Adam
        lr = data['lr']
        print("Start training")
        history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
        #plot_accuracies(history,f"{Path(__file__).parent}")
    
        test_dl = DeviceDataLoader(test_dl, DEVICE)
        eval_results = evaluate(model, test_dl)
        chkp_name = 'last_chkp.pth'
        torch.save(model, chkp_name)

        return jsonify({'train-history': history,"eval-phase":eval_results}), 200
    
    return app

    

if __name__ == '__main__':
    app = flask_service()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)