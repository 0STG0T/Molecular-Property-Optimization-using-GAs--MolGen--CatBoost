from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import pandas as pd
from catboost import CatBoostRegressor

class CatLgKPredictor:
    
    def __init__(self, model_path) -> None:
        self.reg = CatBoostRegressor().load_model(model_path)

    def calculate_descriptors_and_fingerprints(self, mol):
        if not mol:
            descriptors = {desc[0]: None for desc in Descriptors.descList}
            fingerprints = [None] * 1024
        else:
            descriptors = {desc_name: desc_func(mol) for desc_name, desc_func in Descriptors.descList}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints = list(map(int, fp))
        return {**descriptors, "fingerprints": fingerprints}
        
    def preprocess_data(self, data: pd.DataFrame):
        data = data.copy()
       
        data['Molecules'] = data['smiles'].apply(Chem.MolFromSmiles)
        
   
        descriptors_list = data['Molecules'].apply(self.calculate_descriptors_and_fingerprints).tolist()
        
        descriptors_df = pd.DataFrame([d for d in descriptors_list])
        fingerprints_df = pd.DataFrame(descriptors_df.pop('fingerprints').tolist())
        
  
        combined_data = pd.concat([data, descriptors_df, fingerprints_df], axis=1).drop(columns=["smiles", "Molecules"])
        
        return combined_data
        
    def predict_lgK(self, data: pd.DataFrame):
        prep_data = self.preprocess_data(data=data)
    
        preds = self.reg.predict(prep_data)
        
        return preds
