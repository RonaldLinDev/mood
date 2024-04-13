from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import yolov8

ontology = CaptionOntology({"person" : "person",
                            "laptop" : "laptop",
                            "chair" : "chair",
                            "table" : "table",
                            "backpack" : "backpack"})

base_model = GroundedSAM(ontology = ontology)
base_model.label(
  input_folder="./imgas/humandetection/0",
  output_folder="./dataset",
  extension='.png'
)