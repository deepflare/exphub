from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from loguru import logger

from exphub.download.experiment import Experiment

class RelaunchTrigger(ABC):
    """Abstract base class for relaunch triggers.

    A relaunch trigger is used to determine whether an experiment should be relaunched based on certain conditions.
    """
    def filter(self, experiment: Experiment) -> bool:
        """Filter experiments based on the conditions defined in the derived class.

        Args:
            experiment (Experiment): The experiment to be filtered.

        Returns:
            bool: True if the experiment meets the conditions, False otherwise.
        """
        return experiment.filter_via_hyperparams(self.conditions)
    
    @property
    @abstractmethod
    def conditions(self) -> List[str]:
        """Abstract property that should be implemented in derived classes.

        Returns:
            List[str]: A list of conditions that the experiment must meet to be relaunched.
        """
        raise NotImplementedError
    
class FailedStatus(RelaunchTrigger):
    """Relaunch trigger for experiments that have failed."""
    @property
    def conditions(self) -> List[str]:
        """Conditions for relaunching experiments that have failed.

        Returns:
            List[str]: A list of conditions that the experiment must meet to be relaunched.
        """
        return [
            lambda df: df['failed'] == 'True'
        ]
    
class NotEnoughEpochs(RelaunchTrigger):
    """Relaunch trigger for experiments that have not reached a minimum number of epochs."""
    def __init__(self, min_epochs: int) -> None:
        """Initialize the trigger with a minimum number of epochs.

        Args:
            min_epochs (int): The minimum number of epochs required for an experiment.
        """
        self.min_epochs = min_epochs
    @property
    def conditions(self) -> List[str]:
        """Conditions for relaunching experiments that have not reached a minimum number of epochs.

        Returns:
            List[str]: A list of conditions that the experiment must meet to be relaunched.
        """
        return [
            lambda df: df['epoch'] < self.min_epochs
        ]

def _load_default_triggers() -> List[RelaunchTrigger]:
    """Load the default relaunch triggers.

    Returns:
        List[RelaunchTrigger]: A list of default relaunch triggers.
    """
    return [
        FailedStatus(),
        NotEnoughEpochs(5)
    ]
    
@dataclass
class ValidationResult:
    """Data class for storing the results of experiment validation."""
    ids_success: List[str] # all triggers passed
    ids_failed: List[str] # at least one trigger failed    
    
    def summary(self):
        """Print a summary of the validation results."""
        logger.info(f'Validation summary:')
        logger.info(f'* {len(self.ids_success)} runs passed all triggers')
        logger.info(f'* {len(self.ids_failed)} runs failed at least one trigger')
        logger.info(f'* {len(self.ids_success) + len(self.ids_failed)} runs total')
        
        if len(self.ids_failed) == 0:
            logger.success(f'* {len(self.ids_success) / (len(self.ids_success) + len(self.ids_failed)) * 100:.2f}% success rate')
            logger.success(f'* {len(self.ids_failed) / (len(self.ids_success) + len(self.ids_failed)) * 100:.2f}% failure rate')
        else:
            logger.critical(f'* {len(self.ids_success) / (len(self.ids_success) + len(self.ids_failed)) * 100:.2f}% success rate')
            logger.critical(f'* {len(self.ids_failed) / (len(self.ids_success) + len(self.ids_failed)) * 100:.2f}% failure rate')

class Validator:
    """Validator class.

    This class is used to relaunch experiments that have crashed.
    """
    def __init__(self, relaunch_triggers: List[RelaunchTrigger] = _load_default_triggers()) -> None:
        """Initialize the validator with a list of relaunch triggers.

        Args:
            relaunch_triggers (List[RelaunchTrigger], optional): A list of relaunch triggers. Defaults to _load_default_triggers().
        """
        self.relaunch_triggers = relaunch_triggers
        
    def validate(self, experiment):
        """Validate an experiment based on the relaunch triggers.

        Args:
            experiment (Experiment): The experiment to be validated.

        Returns:
            ValidationResult: The results of the validation.
        """
        logger.info(f'Validating experiment')
        experiment_ids = experiment.params[experiment.id_column_name].unique()
        
        trigger2experiment = {
            type(trigger).__name__: trigger.filter(experiment) for trigger in self.relaunch_triggers
        }
        all_failed_ids = []
        
        for trigger_name, trigger_result in trigger2experiment.items():
            if len(trigger_result) == 0:
                logger.success(f'Experiment is empty after applying following trigger: {trigger_name}')
                continue
            ids_failed = trigger_result.params[experiment.id_column_name].unique()
            for id in ids_failed:
                logger.error(f'Run {id} failed trigger {trigger_name}')
            
            all_failed_ids.extend(ids_failed)
            
        all_failed_ids = set(all_failed_ids)        
        ids_success = [id for id in experiment_ids if id not in all_failed_ids]
        
        return ValidationResult(ids_success, ids_failed)
