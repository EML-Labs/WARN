import argparse
from Configurations.Types import TestTypes
from Tests.MimicPerform.ECGTest import MIMICPerformECGTest
from Tests.MimicPerform.PPGTest import MIMICPerformPPGTest

class Manager:
    
    def __init__(self):
        parser = argparse.ArgumentParser(description="Manage tests for WARN project.")
        parser.add_argument('--test_type','-t', type=str, required=True, choices=[test_type.value for test_type in TestTypes], help="Type of test to run.")
        parser.add_argument('--segment_size', '-s', type=int, default=30, help="Segment size for preprocessing.")
        parser.add_argument('--overlap', '-o', type=int, default=5, help="Overlap size for preprocessing.")
        parser.add_argument('--quality_threshold', '-q', type=float, default=0.8, help="Quality threshold for preprocessing.")
        parser.add_argument('--shuffle_data', '-f', type=lambda x: x.lower() == 'true', default=True, help="Whether to shuffle data during testing. Use 'true' or 'false'.")
        parser.add_argument('--batch_size', '-b', type=int, default=32, help="Batch size for testing.")
        self.args = parser.parse_args()

    def run(self):
        match self.args.test_type:
        
            case TestTypes.MIMIC_PERFORM_ECG.value:
                test = MIMICPerformECGTest(
                    segment_size=self.args.segment_size,
                    overlap=self.args.overlap,
                    quality_threshold=self.args.quality_threshold,
                    shuffle_data=self.args.shuffle_data,
                    batch_size=self.args.batch_size
                )
                test.run_test()
            case TestTypes.MIMIC_PERFORM_PPG.value:
                test = MIMICPerformPPGTest(
                    segment_size=self.args.segment_size,
                    overlap=self.args.overlap,
                    quality_threshold=self.args.quality_threshold,
                    shuffle_data=self.args.shuffle_data,
                    batch_size=self.args.batch_size
                )
                test.run_test()
            
            case _:
                raise ValueError(f"Unsupported test type: {self.args.test_type}")