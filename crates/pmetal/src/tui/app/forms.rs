//! Every tab that runs a job from a form, reached through one trait.

use crate::tui::tabs::{FormTab, OllamaTab, ServeTab, Tab, ToolTab, TrainerTab};

/// The form tabs. Held apart from the rest of the app so drawing one can
/// borrow the dashboard's data at the same time.
pub struct Forms {
    pub training: TrainerTab,
    pub embed_train: TrainerTab,
    pub pretrain: TrainerTab,
    pub distillation: TrainerTab,
    pub rlkd: TrainerTab,
    pub grpo: TrainerTab,
    pub preference: TrainerTab,
    pub tokenize: ToolTab,
    pub dflash: ToolTab,
    pub serve: ServeTab,
    pub quantize: ToolTab,
    pub merge: ToolTab,
    pub bench: crate::tui::tabs::BenchTab,
    pub eval: ToolTab,
    pub ollama: OllamaTab,
}

impl Forms {
    pub fn new() -> Self {
        Self {
            training: TrainerTab::training(),
            embed_train: TrainerTab::embed_train(),
            pretrain: TrainerTab::pretrain(),
            distillation: TrainerTab::distillation(),
            rlkd: TrainerTab::rlkd(),
            grpo: TrainerTab::grpo(),
            preference: TrainerTab::preference(),
            tokenize: ToolTab::tokenize(),
            dflash: ToolTab::dflash(),
            serve: ServeTab::new(),
            quantize: ToolTab::quantize(),
            merge: ToolTab::merge(),
            bench: crate::tui::tabs::BenchTab::new(),
            eval: ToolTab::eval(),
            ollama: OllamaTab::new(),
        }
    }

    pub fn get(&self, tab: Tab) -> Option<&dyn FormTab> {
        Some(match tab {
            Tab::Training => &self.training,
            Tab::EmbedTrain => &self.embed_train,
            Tab::Pretrain => &self.pretrain,
            Tab::Distillation => &self.distillation,
            Tab::Rlkd => &self.rlkd,
            Tab::Grpo => &self.grpo,
            Tab::Preference => &self.preference,
            Tab::Tokenize => &self.tokenize,
            Tab::Dflash => &self.dflash,
            Tab::Serve => &self.serve,
            Tab::Quantize => &self.quantize,
            Tab::Merge => &self.merge,
            Tab::Bench => &self.bench,
            Tab::Eval => &self.eval,
            Tab::Ollama => &self.ollama,
            Tab::Dashboard
            | Tab::Device
            | Tab::Models
            | Tab::Datasets
            | Tab::Inference
            | Tab::Jobs => {
                return None;
            }
        })
    }

    pub fn get_mut(&mut self, tab: Tab) -> Option<&mut dyn FormTab> {
        Some(match tab {
            Tab::Training => &mut self.training,
            Tab::EmbedTrain => &mut self.embed_train,
            Tab::Pretrain => &mut self.pretrain,
            Tab::Distillation => &mut self.distillation,
            Tab::Rlkd => &mut self.rlkd,
            Tab::Grpo => &mut self.grpo,
            Tab::Preference => &mut self.preference,
            Tab::Tokenize => &mut self.tokenize,
            Tab::Dflash => &mut self.dflash,
            Tab::Serve => &mut self.serve,
            Tab::Quantize => &mut self.quantize,
            Tab::Merge => &mut self.merge,
            Tab::Bench => &mut self.bench,
            Tab::Eval => &mut self.eval,
            Tab::Ollama => &mut self.ollama,
            Tab::Dashboard
            | Tab::Device
            | Tab::Models
            | Tab::Datasets
            | Tab::Inference
            | Tab::Jobs => {
                return None;
            }
        })
    }

    /// Every form tab, in the order the tab bar lists them.
    pub fn each(&self) -> impl Iterator<Item = (Tab, &dyn FormTab)> {
        Tab::ALL
            .iter()
            .filter_map(|&tab| self.get(tab).map(|form| (tab, form)))
    }

    /// Visit every form tab.
    pub fn for_each_mut(&mut self, mut visit: impl FnMut(Tab, &mut dyn FormTab)) {
        for &tab in Tab::ALL {
            if let Some(form) = self.get_mut(tab) {
                visit(tab, form);
            }
        }
    }
}

impl Default for Forms {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_form_tab_is_reachable_and_the_others_are_not() {
        let mut forms = Forms::new();
        for &tab in Tab::ALL {
            assert_eq!(forms.get(tab).is_some(), tab.is_form(), "{tab}");
            assert_eq!(forms.get_mut(tab).is_some(), tab.is_form(), "{tab}");
        }
        assert_eq!(
            forms.each().count(),
            Tab::ALL.iter().filter(|t| t.is_form()).count()
        );
    }
}
