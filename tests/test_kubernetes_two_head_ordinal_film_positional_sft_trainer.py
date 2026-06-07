import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_film_module():
    module_name = "test_train_kubernetes_two_head_ordinal_film_positional_sft"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "scripts" / "train_kubernetes_two_head_ordinal_film_positional_sft.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("Could not load FiLM positional two-head SFT trainer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TinyBackbone:
    def __init__(self):
        import torch

        class _Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=False)
                self.embed = torch.nn.Embedding(300, 4)
                self.lm = torch.nn.Linear(4, 300)

            def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, return_dict=True):
                hidden = self.embed(input_ids)
                logits = self.lm(hidden)
                loss = logits[..., 0].mean() * 0.0 + 0.7 if labels is not None else None
                return SimpleNamespace(loss=loss, logits=logits, hidden_states=(hidden,))

            def generate(self, *args, **kwargs):  # pragma: no cover
                raise NotImplementedError

        self.module = _Backbone()


class KubernetesTwoHeadOrdinalFilmPositionalSFTTrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.film = load_film_module()

    def test_film_position_features_are_causal_and_shape_stable(self) -> None:
        import torch

        head = self.film.OrdinalLevelHead(
            hidden_size=4,
            hidden_dims=(5, 3),
            dropouts=(0.0, 0.0),
            level_class_count=9,
            level_weights=[1.0] * 9,
            position_encoding="sinusoidal_absolute",
            position_dim=4,
            position_frequencies=(1.0, 2.0),
            position_injection="film_after_512",
            film_hidden_dim=7,
            film_identity_scale=0.1,
        ).module
        line_positions = torch.tensor([[0, 1, 9], [2, 3, 4]], dtype=torch.long)

        features = head.position_features(line_positions, dtype=torch.float32, device=line_positions.device)

        self.assertEqual(tuple(features.shape), (2, 3, 4))
        self.assertTrue(torch.allclose(features[0, 0], torch.tensor([0.0, 1.0, 0.0, 1.0]), atol=1e-6))

    def test_film_identity_initialization_does_not_make_position_change_z(self) -> None:
        import torch

        head = self.film.OrdinalLevelHead(
            hidden_size=4,
            hidden_dims=(5, 3),
            dropouts=(0.0, 0.0),
            level_class_count=9,
            level_weights=[1.0] * 9,
            position_encoding="sinusoidal_absolute",
            position_dim=4,
            position_frequencies=(1.0, 2.0),
            position_injection="film_after_512",
            film_hidden_dim=7,
            film_identity_scale=0.1,
        ).module
        head.eval()
        selected_hidden = torch.randn(1, 3, 4)

        outputs_a = head(selected_hidden, line_positions=torch.tensor([[0, 1, 2]], dtype=torch.long))
        outputs_b = head(selected_hidden, line_positions=torch.tensor([[20, 21, 22]], dtype=torch.long))

        self.assertTrue(torch.allclose(outputs_a.z, outputs_b.z, atol=1e-6))
        self.assertEqual(tuple(outputs_a.ordinal_logits.shape), (1, 3, 8))
        self.assertEqual(tuple(outputs_a.level_logits.shape), (1, 3, 9))
        self.assertEqual(tuple(outputs_a.predicted_levels.shape), (1, 3))

    def test_model_forward_returns_same_public_shapes_as_ordinal_head(self) -> None:
        import torch

        model = self.film.TwoHeadOrdinalQwenForSFT(
            TinyBackbone().module,
            hidden_size=4,
            level_class_count=9,
            ordinal_head_hidden_dims=(5, 3),
            ordinal_head_dropouts=(0.0, 0.0),
            level_weights=[1.0] * 9,
            lambda_level=2.0,
            position_encoding="sinusoidal_absolute",
            position_dim=4,
            position_frequencies=(1.0, 2.0),
            position_injection="film_after_512",
            film_hidden_dim=7,
            film_identity_scale=0.1,
        ).module
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        labels = torch.tensor([[-100, 2, 3, 4]], dtype=torch.long)
        positions = torch.tensor([[1, 3]], dtype=torch.long)
        line_positions = torch.tensor([[0, 1]], dtype=torch.long)
        level_labels = torch.tensor([[0, 6]], dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            level_label_positions=positions,
            level_line_positions=line_positions,
            level_labels=level_labels,
        )

        self.assertIsNotNone(outputs.loss)
        self.assertEqual(tuple(outputs.ordinal_z.shape), (1, 2))
        self.assertEqual(tuple(outputs.ordinal_logits.shape), (1, 2, 8))
        self.assertEqual(tuple(outputs.level_logits.shape), (1, 2, 9))
        self.assertEqual(tuple(outputs.predicted_levels.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
