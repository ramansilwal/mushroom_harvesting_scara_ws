
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def forward_kinematics_from_angles(theta1,theta2,theta3):
    r1= 0.325
    r2 = 0.275
    r3=0.21
    x1 = r1 * np.cos(theta1)
    x2= x1 + r2 * np.cos(theta1 + theta2)
    x3 = x2 + r3 * np.cos(theta1 + theta2 + theta3)

    y1 = r1 * np.sin(theta1)
    y2 = y1 + r2 * np.sin(theta1 + theta2)
    y3 = y2 + r3 * np.sin(theta1 + theta2 + theta3)


    return [[0,0,],[x1,y1],[x2,y2],[x3,y3]]

r1= 0.325
r2 = 0.275
r3=0.175
class MDNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_gaussians):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.out_features = out_features

        self.pi = nn.Linear(in_features, num_gaussians)
        self.sigma = nn.Linear(in_features, num_gaussians * out_features)
        self.mu = nn.Linear(in_features, num_gaussians * out_features)

    def forward(self, x):
        pi = F.softmax(self.pi(x), dim=1)
        sigma = F.softplus(self.sigma(x)) + 1e-6
        mu = self.mu(x)

        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = mu.view(-1, self.num_gaussians, self.out_features)

        return pi, sigma, mu

class IK_MDN_Model(nn.Module):
    def __init__(self, input_size=2, output_size=3, num_gaussians=50, dropout_prob=0.1):
        super().__init__()
        self.hidden =  nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mdn = MDNLayer(in_features=128, out_features=output_size, num_gaussians=num_gaussians)

    def forward(self, x):
        features = self.hidden(x)
        pi, sigma, mu = self.mdn(features)
        return pi, sigma, mu
    
    def sample_from_mdn(self,pi, sigma, mu, num_samples=10_000):
        """Sample joint angles from MDN output."""
        n = pi.size(0)
        pi_np = pi.detach().cpu().numpy()
        sigma_np = sigma.detach().cpu().numpy()
        mu_np = mu.detach().cpu().numpy()
        print(pi_np)
        samples = []

        for i in range(n):
            sample_list = []
            for _ in range(num_samples):
                
                component = np.random.choice(len(pi_np[i]), p=pi_np[i])
                sample = np.random.normal(loc=mu_np[i, component], scale=sigma_np[i, component])
                sample_list.append(sample)
            samples.append(sample_list)

        return np.array(samples) 
    
    def give_ik(self,goal,current_angle,device="cuda"):
        current_angle=[i if i!=0 else 0.00001 for i in current_angle ]
        goal_batch = torch.tensor([goal], dtype=torch.float32,device=device)
        pi, sigma, mu = self.__call__(goal_batch)
        samples=self.sample_from_mdn(pi, sigma, mu)
        valid_angles=[]
        similaritys=[]
        if len(samples)>0:       
            for angle1,angle2,angle3 in samples[0]:
                ee=np.array(forward_kinematics_from_angles(angle1,angle2,angle3)[-1])
                goal_np=np.array(goal[0])
                if np.linalg.norm(ee-goal)<0.0075:
                    similarity = cosine_similarity(np.array([[angle1,angle2,angle3]]), np.array([current_angle]))
                    valid_angles.append([angle1,angle2,angle3])
                    similaritys.append(similarity[0][0])
            if len(similaritys)>1:
                return valid_angles[similaritys.index(max(similaritys))]
        else:
            return None
        return None

    def give_ik_degree(self,goal,current_angle,device="cuda"):
        val=self.give_ik(goal,[np.deg2rad(i) for i in current_angle],device=device)
        if val is not None:
            return [int(np.rad2deg(i)) for i in val]
        
if __name__=="__main__":
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IK_MDN_Model(input_size=2, output_size=3, num_gaussians=50, dropout_prob=0.1)
    model.load_state_dict(torch.load("/home/cotrobot/gripper/models/ik_mdn_final_model_50.pth"))
    model.eval()
    model.to(device=device)
    print(model.give_ik_degree([0.75,0.05],[0,0,0]))