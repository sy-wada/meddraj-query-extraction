import torch
from torchcrf import CRF
import torch.nn as nn
from transformers import (
    AutoConfig,
    ModernBertModel,
    ModernBertPreTrainedModel,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead
from typing import Optional, List, Tuple, Union, Any
from transformers.modeling_outputs import TokenClassifierOutput

def create_transitions(
    label2id: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """遷移スコアを定義"""
    # "B-"のラベルIDのlist
    b_ids = [v for k, v in label2id.items() if k[0] == "B"]
    # I-のラベルIDのlist
    i_ids = [v for k, v in label2id.items() if k[0] == "I"]
    o_id = label2id["O"]  # OのラベルID

    # 開始遷移スコアを定義する
    # すべてのスコアを-100で初期化する
    start_transitions = torch.full([len(label2id)], -100.0)
    # "B-"のラベルへ遷移可能として0を代入する
    start_transitions[b_ids] = 0
    # "O"のラベルへ遷移可能として0を代入する
    start_transitions[o_id] = 0

    # ラベル間の遷移スコアを定義する
    # すべてのスコアを-100で初期化する
    transitions = torch.full([len(label2id), len(label2id)], -100.0)
    # すべてのラベルから"B-"へ遷移可能として0を代入する
    transitions[:, b_ids] = 0
    # すべてのラベルから"O"へ遷移可能として0を代入する
    transitions[:, o_id] = 0
    # "B-"から同じタイプの"I-"へ遷移可能として0を代入する
    transitions[b_ids, i_ids] = 0
    # "I-"から同じタイプの"I-"へ遷移可能として0を代入する
    transitions[i_ids, i_ids] = 0

    # 終了遷移スコアを定義する
    # すべてのラベルから遷移可能としてすべてのスコアを0とする
    end_transitions = torch.zeros(len(label2id))
    return start_transitions, transitions, end_transitions

class MultiTaskModernBertCRF(ModernBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_tasks = config.num_tasks
        
        # 1) 共有エンコーダ
        self.encoder   = ModernBertModel(config)
        h = config.hidden_size

        # 2) CLS ヘッド
        # self.cls_head  = nn.Linear(h, config.num_labels_cls)
        self.cls1_head = ModernBertPredictionHead(config)
        self.cls1_drop = torch.nn.Dropout(config.classifier_dropout)
        self.cls1_classifier = nn.Linear(config.hidden_size, config.num_labels_cls1)

        # 3) NER-1 ヘッド + CRF
        self.ner1_fc   = nn.Linear(h, config.num_labels_ner1)
        self.crf1      = CRF(config.num_labels_ner1, batch_first=True)

        # 4) NER-2 ヘッド + CRF
        self.ner2_fc   = nn.Linear(h, config.num_labels_ner2)
        self.crf2      = CRF(config.num_labels_ner2, batch_first=True)

        # 5) 重み初期化
        self.init_weights()

        # 6) BIO 制約付き初期化
        #    create_transitions は label2id → (start, trans, end) を返すヘルパー
        with torch.no_grad():
            st1, t1, et1 = create_transitions(config.label2id_ner1)
            self.crf1.start_transitions.copy_(st1)
            self.crf1.transitions.copy_(t1)
            self.crf1.end_transitions.copy_(et1)

            st2, t2, et2 = create_transitions(config.label2id_ner2)
            self.crf2.start_transitions.copy_(st2)
            self.crf2.transitions.copy_(t2)
            self.crf2.end_transitions.copy_(et2)
        
        # task_weights を保持するためのバッファを登録 (永続的ではないが、forwardで使える)
        self.register_buffer('current_task_weights', torch.ones(self.num_tasks), persistent=False) 
    
    def set_task_weights(self, task_weights_tensor: torch.Tensor):
        # deviceを合わせる必要があるかもしれない
        self.current_task_weights = task_weights_tensor.to(self.current_task_weights.device)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                special_tokens_mask=None,
                labels_cls1=None,
                labels_ner1=None,
                labels_ner2=None,
                # task_weights=None,
                alpha: Optional[float] = None,
                gamma: float = 2.0):
        
        # 引数からではなく、バッファからtask_weightsを取得
        # if task_weights is None:
        #     task_weights = self.current_task_weights
        task_weights = self.current_task_weights
        # print(f"Debug in Model Forward: Using task_weights from buffer: {task_weights}, shape: {task_weights.shape}")

        # 1) フォールバック：alpha が None なら 1.0
        if alpha is None:
            alpha = 1.0

        # print(f"Debug in Model Forward: Received task_weights: {task_weights}, shape: {task_weights.shape if hasattr(task_weights, 'shape') else 'N/A (not a tensor or no shape)'}") # デバッグプリント追加

        # 2) エンコーダ
        enc_out = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = enc_out.last_hidden_state
        # pooler_output = enc_out.pooler_output # CLS用 (もしModernBertModelがpooler_outputを返せば)

        # 3) 各ヘッドのロジット

        # CLS-1
        if self.config.classifier_pooling == "cls":
            cls_input_tensor = sequence_output[:, 0]
        elif self.config.classifier_pooling == "mean":
            if attention_mask is None: # attention_maskがNoneの場合のエラーを防ぐ
                raise ValueError("attention_mask must be provided for mean pooling")
            cls_input_tensor = (sequence_output * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )
        else: # デフォルトまたは他のプーリング戦略
            # ModernBertModelがpooler_outputを返す場合、それを使用するのが一般的
            # もし返さない場合、sequence_output[:, 0] を使うことが多い
            # ここでは、 config.classifier_pooling の値に応じて分岐しているため、
            # 'cls' と 'mean' 以外の場合の処理が必要であれば追加する
            # 例として、 'cls' と同じ処理にするか、エラーを出す
            cls_input_tensor = sequence_output[:, 0] # フォールバックとしてCLSトークンを使用

        pooled_output = self.cls1_head(cls_input_tensor)
        pooled_output = self.cls1_drop(pooled_output)
        logit_cls1  = self.cls1_classifier(pooled_output)

        # NER-1
        logit_ner1 = self.ner1_fc(sequence_output)

        # NER-2
        logit_ner2 = self.ner2_fc(sequence_output)

        # 4) 損失計算
        losses = []
        # unweighted_task_losses = [None] * 3 # Assuming 3 tasks, align with num_tasks later
        unweighted_task_losses: list[Optional[torch.Tensor]] = [None] * self.num_tasks 
        # --- CLS: FocalLoss ---
        if labels_cls1 is not None:
            # labels_cls1 は (batch_size, num_classes) で、-100.0 でパディングされている可能性がある
            # -100.0 でパディングされたサンプルを除外するためのマスクを作成
            # (行全体が -100.0 かどうかをチェック)
            valid_cls_samples_mask = ~torch.all(labels_cls1 == -100.0, dim=1)
            
            if torch.any(valid_cls_samples_mask): # 有効なサンプルが1つでもあれば損失を計算
                valid_labels_cls1 = labels_cls1[valid_cls_samples_mask]
                # print(f"{logit_cls1.device}, {valid_cls_samples_mask.device}")
                # Ensure the mask is on the same device as the logits
                valid_cls_samples_mask = valid_cls_samples_mask.to(logit_cls1.device)
                valid_logit_cls1 = logit_cls1[valid_cls_samples_mask]

                probs = torch.sigmoid(valid_logit_cls1)
                p_t   = torch.where(valid_labels_cls1 == 1, probs, 1 - probs)
                focal_loss_unweighted_per_sample = -alpha * (1 - p_t) ** gamma * torch.log(p_t.clamp_min(1e-9))
                
                # 複数のクラスがある場合、各サンプルのクラスごとの損失を平均する (または合計する)
                # ここでは mean() を使って、サンプル内のクラス間で平均し、その後バッチで平均する
                if focal_loss_unweighted_per_sample.ndim > 1:
                     focal_loss_unweighted = focal_loss_unweighted_per_sample.mean(dim=1).mean() # クラス平均 -> バッチ平均
                else: # 1クラス分類の場合 (通常マルチラベルでは発生しない)
                     focal_loss_unweighted = focal_loss_unweighted_per_sample.mean() # バッチ平均
                
                if focal_loss_unweighted.requires_grad:
                     unweighted_task_losses[0] = focal_loss_unweighted
                losses.append(task_weights[0] * focal_loss_unweighted)
            else:
                # 有効なCLSサンプルがない場合、損失は0とするか、あるいは unweighted_task_losses[0] を None のままにする
                # GradNormのために、勾配を持つ0テンソルを unweighted_task_losses に設定することを検討
                # ただし、losses には何も追加しないか、あるいは task_weights[0] * 0.0 を追加する
                # unweighted_task_losses[0] = torch.tensor(0.0, device=logit_cls1.device, requires_grad=True) # 例
                pass # unweighted_task_losses[0] は None のまま

        # --- NER-1: CRF ---
        if labels_ner1 is not None:
            loss1_value = torch.tensor(0.0, device=logit_ner1.device)
            valid_crf_indices_ner1 = torch.where(labels_ner1[:, 1] != -100)[0]
            if len(valid_crf_indices_ner1) > 0:
                # Extract data for valid samples
                logit_ner1_valid = logit_ner1[valid_crf_indices_ner1, 1:, :]
                
                labels_ner1_internal = labels_ner1.clone() # Clone to avoid modifying original
                labels_ner1_internal_valid = labels_ner1_internal[valid_crf_indices_ner1, 1:]
                
                # Create mask for the valid sub-batch (should satisfy CRF validation)
                mask_ner1_valid = (labels_ner1_internal_valid != -100).bool()
                
                # マスクがFalseの位置のラベルを0で埋める (範囲外アクセスを防ぐため)
                # CRFはマスクがTrueの箇所のみを考慮するため、損失計算には影響しない
                labels_ner1_internal_valid_masked = torch.where(
                    mask_ner1_valid, labels_ner1_internal_valid, torch.zeros_like(labels_ner1_internal_valid)
                )
                
                loss1_crf_unweighted = -self.crf1( # This is the unweighted loss for the task
                    emissions=logit_ner1_valid,
                    tags=labels_ner1_internal_valid_masked,
                    mask=mask_ner1_valid,
                    reduction="mean"
                )
                loss1_value = loss1_crf_unweighted
                if loss1_value.requires_grad: # Store if it's a valid loss
                    unweighted_task_losses[1] = loss1_value
            losses.append(task_weights[1] * loss1_value)

        # --- NER-2: CRF ---
        if labels_ner2 is not None:
            loss2_value = torch.tensor(0.0, device=logit_ner2.device)
            valid_crf_indices_ner2 = torch.where(labels_ner2[:, 1] != -100)[0]
            if len(valid_crf_indices_ner2) > 0:
                logit_ner2_valid = logit_ner2[valid_crf_indices_ner2, 1:, :]
                labels_ner2_internal = labels_ner2.clone()
                labels_ner2_internal_valid = labels_ner2_internal[valid_crf_indices_ner2, 1:]
                mask_ner2_valid = (labels_ner2_internal_valid != -100).bool()

                # マスクがFalseの位置のラベルを0で埋める
                labels_ner2_internal_valid_masked = torch.where(
                    mask_ner2_valid, labels_ner2_internal_valid, torch.zeros_like(labels_ner2_internal_valid)
                )
                
                loss2_crf_unweighted = -self.crf2( # This is the unweighted loss for the task
                    emissions=logit_ner2_valid,
                    tags=labels_ner2_internal_valid_masked,
                    mask=mask_ner2_valid,
                    reduction="mean"
                )
                loss2_value = loss2_crf_unweighted
                if loss2_value.requires_grad: # Store if it's a valid loss
                    unweighted_task_losses[2] = loss2_value
            losses.append(task_weights[2] * loss2_value)

        # 5) 合計損失
        # Filter out None from losses before stacking, if any task had no valid samples
        valid_losses = [l for l in losses if l is not None and l.requires_grad]
        loss = torch.stack(valid_losses).sum() if valid_losses else torch.tensor(0.0, device=self.device, requires_grad=True)

        # 6) 出力フォーマット
        return {
            "loss":       loss,
            "unweighted_task_losses": unweighted_task_losses,
            "logits_cls1": logit_cls1,
            "logits_ner1": logit_ner1,
            "logits_ner2": logit_ner2,
            "hidden_states": enc_out.hidden_states,
            "attentions":    enc_out.attentions,
            "labels_cls1": labels_cls1,
            "labels_ner1": labels_ner1,
            "labels_ner2": labels_ner2,
        }
    
class MultiTaskDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        super().__init__(tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    def __call__(self, features):
        if not features: # features が空のリストの場合
            return {}

        # # --- デバッグプリント ---
        # if features: # featuresが空でないことを確認
        #     print(f"DataCollator input feature 0 type: {type(features[0])}")
        #     for i, f_item in enumerate(features[:2]): # 最初の2サンプル程度をチェック
        #         print(f"  Feature {i}:")
        #         for key, val in f_item.items():
        #             if "label" in key: # ラベル関連のキーのみ表示
        #                 print(f"    {key}: type={type(val)}, value (or shape if tensor)={val if not hasattr(val, 'shape') else val.shape}")
        # # --- デバッグプリントここまで ---

        # 1) CLS/NER ラベルを分離 (この時点ではPythonのリストの可能性がある)
        original_cls1_labels_list = [f.pop("labels_cls1", None) for f in features]
        ner1_labels_list = [f.pop("labels_ner1", None) for f in features]
        ner2_labels_list = [f.pop("labels_ner2", None) for f in features]

        # 2) 入力パディング (input_ids, attention_mask, etc.)
        # features には "input_ids", "attention_mask", "special_tokens_mask" のみが残っている想定
        batch = super().__call__(features)

        # 基準デバイスを取得 (通常は input_ids が存在するデバイス)
        # batchが空でないこと、およびinput_idsが存在することを確認
        if not batch or "input_ids" not in batch or batch["input_ids"] is None:
            # このケースは通常、featuresが空の場合か、予期せぬ入力の場合
            # 空のバッチや必要なキーがない場合は、そのまま返すかエラー処理
            # ここでは、以降の処理でデバイスが必要なため、安全に空の辞書を返すか、
            # あるいはデフォルトデバイスを設定する (例: torch.device("cpu"))
            # ただし、上位の処理で空のfeaturesが渡された場合は、ここで {} が返されるため、
            # ここまで到達する場合は batch には何らかのキーが存在するはず。
            # もし input_ids がなければ、他のテンソルからデバイスを取得するか、
            # デフォルトデバイスを使用する。
            # ここでは、input_ids が存在することを前提とする。なければエラー。
            if not features: # featuresが本当に空なら、最初のifで処理済みのはず
                return {}
            # features が空でなく、batch に input_ids がない場合は異常系
            # device = torch.device("cpu") # フォールバックデバイス
            # print("Warning: 'input_ids' not found in batch from super().__call__(). Using CPU as default device for DataCollator.")
            # しかし、通常super().__call__(features)はinput_idsを含むので、このパスは稀
            # ここでエラーにするか、安全なデフォルトを設定するかは要件による
            # ここでは、input_ids が必ず存在し、Noneでないと仮定して進める。
            # (もしNoneになりうるなら、device取得前にNoneチェックとフォールバックが必要)
            pass # 下のdevice取得に進む。もしbatch["input_ids"]がNoneならエラーになる。

        device = batch["input_ids"].device

        # 3) CLS ラベルを (B, num_classes) テンソルに
        num_cls1_classes = None
        for lbl_item in original_cls1_labels_list:
            if lbl_item is not None:
                if isinstance(lbl_item, list):
                    num_cls1_classes = len(lbl_item)
                elif isinstance(lbl_item, torch.Tensor): # dataset map 後は list になっているはずだが念のため
                    num_cls1_classes = lbl_item.shape[0]
                else:
                    raise TypeError(f"labels_cls1 item has unexpected type: {type(lbl_item)}. Value: {lbl_item}")
                break
        
        if num_cls1_classes is not None: # 少なくとも1つのサンプルにcls1ラベルが存在する場合
            cls1_processed_tensors = []
            for lbl_item in original_cls1_labels_list:
                if lbl_item is not None:
                    # Pythonリストまたはテンソルを適切な型のテンソルに変換
                    if isinstance(lbl_item, list):
                        tensor_lbl = torch.tensor(lbl_item, dtype=torch.int64, device=device)
                    elif isinstance(lbl_item, torch.Tensor):
                        tensor_lbl = lbl_item.to(dtype=torch.int64, device=device) # 型を統一
                    else: # このケースは上のループでキャッチされるはず
                         raise TypeError(f"Unexpected type for cls1 label item: {type(lbl_item)}")
                    
                    if tensor_lbl.shape[0] != num_cls1_classes:
                        raise ValueError(
                            f"Inconsistent number of classes in labels_cls1. "
                            f"Expected {num_cls1_classes}, got {tensor_lbl.shape[0]} for item {lbl_item}"
                        )
                    cls1_processed_tensors.append(tensor_lbl)
                else: # ラベルがNoneの場合
                    cls1_processed_tensors.append(torch.full((num_cls1_classes,), -100, dtype=torch.int64, device=device))
            
            if cls1_processed_tensors: # リストが空でなければスタック
                 batch["labels_cls1"] = torch.stack(cls1_processed_tensors).to(dtype=torch.float).to(device)
        # else: すべてのCLSラベルがNoneだった場合、batch["labels_cls1"] は作成されない。これはモデル側で処理。

        # 4) NER ラベルを (B, T) にパディング
        seq_len = batch["input_ids"].size(1)

        def _pad_ner_labels_internal(label_list_for_batch, target_seq_len, target_device):
            if not label_list_for_batch: # 入力リスト自体が空ならNone (featuresが空のケースなど)
                return None

            padded_tensors = []
            for item in label_list_for_batch: # item は PythonのリストかTensorかNone
                if item is not None:
                    # PythonリストかTensorかを判定して、適切な型のTensorに変換
                    if isinstance(item, list):
                        current_tensor = torch.tensor(item, dtype=torch.long, device=target_device) # device指定追加
                    elif isinstance(item, torch.Tensor):
                        current_tensor = item.to(dtype=torch.long, device=target_device) # device指定追加、型を統一
                    else:
                        raise TypeError(f"NER label item has unexpected type: {type(item)}")
                    
                    current_len = current_tensor.shape[0]
                    if current_len < target_seq_len:
                        padding = torch.full((target_seq_len - current_len,), -100, dtype=torch.long, device=target_device) # device指定追加
                        padded_tensors.append(torch.cat([current_tensor, padding]))
                    elif current_len > target_seq_len: # 通常は発生しづらい (preprocessでクリップされるため)
                        padded_tensors.append(current_tensor[:target_seq_len])
                    else: # current_len == target_seq_len
                        padded_tensors.append(current_tensor)
                else: # item is None
                    padded_tensors.append(torch.full((target_seq_len,), -100, dtype=torch.long, device=target_device)) # device指定追加
            
            return torch.stack(padded_tensors) if padded_tensors else None


        padded_ner1 = _pad_ner_labels_internal(ner1_labels_list, seq_len, device)
        if padded_ner1 is not None:
            batch["labels_ner1"] = padded_ner1

        padded_ner2 = _pad_ner_labels_internal(ner2_labels_list, seq_len, device)
        if padded_ner2 is not None:
            batch["labels_ner2"] = padded_ner2
            
        return batch

class GradNormTrainer(Trainer):
    def __init__(self, *args,
                 num_tasks: int = 3,
                 alpha_gradnorm: float = 1.5,
                 focal_loss_alpha: Optional[float] = None,
                 focal_loss_gamma: float = 2.0,
                 log_vars_learning_rate: float = 0.01,
                 debug_gradnorm: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.alpha_gradnorm = alpha_gradnorm
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.log_vars_learning_rate = log_vars_learning_rate
        self.debug_gradnorm = debug_gradnorm # ★★★ 追加 ★★★
        self.log_vars = torch.nn.Parameter(torch.zeros(self.num_tasks))

    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if p.requires_grad],
                },
                {
                    "params": [self.log_vars],
                    "lr": self.log_vars_learning_rate, 
                    "weight_decay": 0.0
                },
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            
            model_lr = optimizer_kwargs.get("lr", self.args.learning_rate)
            optimizer_grouped_parameters[0]["lr"] = model_lr
            if 'lr' in optimizer_kwargs:
                del optimizer_kwargs['lr']

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        # print(f"Debug GradNormTrainer: compute_loss called. model type: {type(model)}, model device: {model.device if hasattr(model, 'device') else 'N/A'}")
        # if hasattr(model, 'module'):
        #     print(f"Debug GradNormTrainer: model.module type: {type(model.module)}, model.module device: {model.module.device if hasattr(model.module, 'device') else 'N/A'}")
        # else:
        #     print("Debug GradNormTrainer: model.module does not exist. Not wrapped in DataParallel?")

        true_model = model.module if hasattr(model, 'module') else model
        
        # print(f"Debug GradNormTrainer: inputs['input_ids'].device: {inputs.get('input_ids').device if inputs.get('input_ids') is not None else 'N/A'}")
        # if inputs.get('labels_cls1') is not None:
        #     print(f"Debug GradNormTrainer: inputs['labels_cls1'].device: {inputs.get('labels_cls1').device}")

        current_task_weights = self.log_vars.exp()
        true_model.set_task_weights(current_task_weights)

        if self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
            print(f"\n--- GradNorm Debug: Step {self.state.global_step} ---")
            print(f"  Computed task_weights for this forward: {current_task_weights.data.tolist()}")

        # # nn.DataParallel ラッパーの model を直接呼び出す
        # # forward に task_weights を渡す
        # model_inputs = {
        #     "input_ids": inputs.get("input_ids"),
        #     "attention_mask": inputs.get("attention_mask"),
        #     "labels_cls1": inputs.get("labels_cls1"),
        #     "labels_ner1": inputs.get("labels_ner1"),
        #     "labels_ner2": inputs.get("labels_ner2"),
        #     "task_weights": current_task_weights.to(inputs.get("input_ids").device if inputs.get("input_ids") is not None else self.args.device),
        #     "alpha": self.focal_loss_alpha,
        #     "gamma": self.focal_loss_gamma,
        # }
        # model_inputs_filtered = {k: v for k, v in model_inputs.items() if v is not None}
        # outputs = model(**model_inputs_filtered)

        outputs = true_model(
            input_ids      = inputs.get("input_ids"),
            attention_mask = inputs.get("attention_mask"),
            labels_cls1    = inputs.get("labels_cls1"),
            labels_ner1    = inputs.get("labels_ner1"),
            labels_ner2    = inputs.get("labels_ner2"),
            # task_weights 引数は forward から削除したので、ここからも削除
            alpha          = self.focal_loss_alpha,
            gamma          = self.focal_loss_gamma,
        )

        total_loss = outputs.get("loss")
        unweighted_task_losses = outputs.get("unweighted_task_losses")

        if self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0 and unweighted_task_losses is not None:
            print(f"  Unweighted task_losses (device: {unweighted_task_losses[0].device if unweighted_task_losses and unweighted_task_losses[0] is not None else 'N/A'}): {[l.item() if l is not None else 'N/A' for l in unweighted_task_losses]}")

        if total_loss is None or not total_loss.requires_grad or unweighted_task_losses is None:
            # total_loss や unweighted_task_losses は cuda:0 にあるはず
            final_loss = total_loss if total_loss is not None else torch.tensor(0.0, device=self.args.device, requires_grad=True)
            return (final_loss, outputs) if return_outputs else final_loss

        # GradNorm の計算は model.module (true_model) のパラメータに対して行う
        true_model = model.module if hasattr(model, 'module') else model
        shared_encoder_params = [p for p in true_model.encoder.parameters() if p.requires_grad]
        if not shared_encoder_params:
            return (total_loss, outputs) if return_outputs else total_loss
        
        W_grad_target = shared_encoder_params[-1] 
        G_calculated = []
        # W_grad_target が存在するデバイスを取得 (これが基準デバイスとなるべき)
        target_device = W_grad_target.device

        for i in range(self.num_tasks):
            task_loss_i_unweighted = unweighted_task_losses[i]
            w_i = self.log_vars.exp()[i] 
            
            if task_loss_i_unweighted is not None and task_loss_i_unweighted.requires_grad and task_loss_i_unweighted.item() > 0:
                weighted_task_loss_i = w_i * task_loss_i_unweighted
                grad_W_i_tuple = torch.autograd.grad(
                    outputs=weighted_task_loss_i,
                    inputs=W_grad_target,
                    grad_outputs=torch.ones_like(weighted_task_loss_i).to(target_device), # grad_outputsもデバイス指定
                    retain_graph=True, 
                    create_graph=True, # create_graph=True を GradNorm 損失の計算のために追加
                    allow_unused=True
                )
                grad_W_i = grad_W_i_tuple[0]
                if grad_W_i is not None:
                    G_i_val = grad_W_i.norm(2) 
                    G_calculated.append(G_i_val.to(target_device)) # 明示的にデバイス指定
                else:
                    # grad_W_i が None の場合、このタスクの G_i は 0 とするが、log_vars[i]への依存性は保持
                    G_calculated.append((w_i * torch.tensor(0.0, requires_grad=w_i.requires_grad)).to(target_device)) # 明示的にデバイス指定
            else:
                # 損失がないか勾配がないタスクの場合も、形式的にlog_vars[i]に依存する0を追加
                G_calculated.append((w_i * torch.tensor(0.0, requires_grad=w_i.requires_grad)).to(target_device)) # 明示的にデバイス指定
        
        if not G_calculated: # G_calculated が空の場合の処理を追加 (実際にはnum_tasks分0が入るので稀)
            grad_norm_loss = torch.tensor(0.0, device=target_device, requires_grad=True)
        elif not any(gc.requires_grad for gc in G_calculated):
            grad_norm_loss = torch.tensor(0.0, device=target_device, requires_grad=True) # target_device を使用
            
            if self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
                 print(f"  DEBUG: No valid G_calculated or no element in G_calculated has requires_grad=True. Setting grad_norm_loss to 0.")
        else:
            # G_calculated のうち、requires_grad=False のものがあれば、Trueに変換してからstackする
            # これにより、G_tensor全体がrequires_grad=Trueになりやすくなる
            valid_G_for_stacking = []
            for gc_idx, gc_val in enumerate(G_calculated):
                if gc_val.requires_grad:
                    valid_G_for_stacking.append(gc_val)
                else:
                    # requires_grad=False の場合、detach()して新たにTrueにする (計算グラフには影響しないが、L_gradの計算には必要)
                    # ただし、元のG_calculatedの要素がlog_varsに依存していないと、ここでのTrueは意味がない
                    # 基本的にG_calculatedの要素はw_iを通じてlog_varsに依存するはず
                    # print(f"  DEBUG G_calculated[{gc_idx}] was requires_grad=False. Value: {gc_val.item()}")
                    valid_G_for_stacking.append(gc_val.detach().requires_grad_(True)) # これはあまり良くない可能性

            G_tensor = torch.stack(G_calculated) # G_calculated は既に w_i に依存しているはずなのでこれで良い

            
            if self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
                print(f"  G_tensor (grad norms for each task): {G_tensor.data.tolist()}")
                print(f"  DEBUG: G_tensor.requires_grad: {G_tensor.requires_grad}")
                for k_debug, g_val_debug in enumerate(G_calculated):
                    print(f"  DEBUG: G_calculated[{k_debug}].requires_grad: {g_val_debug.requires_grad}, value: {g_val_debug.item()}")
            
            G_avg = G_tensor.mean()
            
            if self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
                print(f"  G_avg (average grad norm): {G_avg.item():.4f}, G_avg.requires_grad: {G_avg.requires_grad}")

            if torch.isclose(G_avg, torch.tensor(0.0, device=G_avg.device)) or G_avg.item() < 1e-8:
                grad_norm_loss = torch.tensor(0.0, device=target_device, requires_grad=True) # target_device を使用
                
                if self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
                     print(f"  GradNorm: G_avg is close to zero. Setting grad_norm_loss to 0.")
            else:
                loss_terms = []
                for i in range(self.num_tasks):
                    g_i = G_tensor[i]
                    relative_grad_strength_i = g_i / (G_avg + 1e-8) 
                    target_grad_norm_factor_i = relative_grad_strength_i ** self.alpha_gradnorm
                    term_i = (g_i - G_avg * target_grad_norm_factor_i).abs()
                    loss_terms.append(term_i)
                    
                    if self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
                         print(f"  Task {i}: G_i={g_i.item():.4f} (req_grad={g_i.requires_grad}), RelStr={relative_grad_strength_i.item():.4f}, TargetFactor={target_grad_norm_factor_i.item():.4f}, Term={term_i.item():.4f} (req_grad={term_i.requires_grad})")
                
                if loss_terms:
                    grad_norm_loss = torch.stack(loss_terms).sum()
                else:
                    grad_norm_loss = torch.tensor(0.0, device=target_device, requires_grad=True) # target_device を使用

        
        if self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
            print(f"  GradNorm Loss (L_grad): {grad_norm_loss.item():.4f}, L_grad.requires_grad: {grad_norm_loss.requires_grad}, L_grad.is_leaf: {grad_norm_loss.is_leaf}")
        
        if self.log_vars.grad is not None:
            
            if self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
                 print(f"  DEBUG: self.log_vars.grad (from total_loss, before zeroing for L_grad): {self.log_vars.grad.data.tolist()}")
            self.log_vars.grad.data.zero_() 
        elif self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
             print(f"  DEBUG: self.log_vars.grad was None (before L_grad calculation)")

        if grad_norm_loss.requires_grad and not grad_norm_loss.is_leaf: # L_grad が計算グラフに接続されている場合のみ逆伝播
            # grad_norm_loss.backward() を直接呼ぶ代わりに、torch.autograd.gradで明示的に勾配を計算
            log_vars_grad = torch.autograd.grad(
                outputs=grad_norm_loss,
                inputs=self.log_vars,
                grad_outputs=torch.ones_like(grad_norm_loss), # grad_norm_loss がスカラーなので
                retain_graph=False, # このパスでの最後の勾配計算なのでFalseで良い
                allow_unused=True # log_vars に grad_norm_loss が依存していないケース（通常はないはず）
            )[0] # .grad は self.log_vars に対する勾配のみを取得

            if log_vars_grad is not None:
                if self.log_vars.grad is None: # まだ勾配バッファがなければ初期化
                    self.log_vars.grad = torch.zeros_like(self.log_vars.data)
                self.log_vars.grad.copy_(log_vars_grad) # 計算された勾配をコピー

                
                if self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
                    print(f"  DEBUG: self.log_vars.grad (manually set from L_grad): {self.log_vars.grad.data.tolist()}")
            elif self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
                print(f"  DEBUG: log_vars_grad calculated from L_grad is None. self.log_vars.grad not updated by L_grad.")

        elif self.debug_gradnorm and self.state.is_local_process_zero and self.state.global_step % self.args.logging_steps == 0:
            print(f"  DEBUG: Skipping manual grad calculation for log_vars. grad_norm_loss.requires_grad: {grad_norm_loss.requires_grad}, grad_norm_loss.is_leaf: {grad_norm_loss.is_leaf}")
        
        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Any, Any]:
        
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            eval_model = model.module if hasattr(model, 'module') else model

            outputs = eval_model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                alpha=self.focal_loss_alpha, 
                gamma=self.focal_loss_gamma, 
            )

            loss = outputs.get("loss")
            
            detached_logits_cls1 = outputs.get("logits_cls1").detach() if outputs.get("logits_cls1") is not None else None
            detached_logits_ner1 = outputs.get("logits_ner1").detach() if outputs.get("logits_ner1") is not None else None
            detached_logits_ner2 = outputs.get("logits_ner2").detach() if outputs.get("logits_ner2") is not None else None
            logits_tuple = (detached_logits_cls1, detached_logits_ner1, detached_logits_ner2)

            labels_cls1_tensor = inputs.get("labels_cls1")
            detached_labels_cls1 = labels_cls1_tensor.detach() if isinstance(labels_cls1_tensor, torch.Tensor) else labels_cls1_tensor

            labels_ner1_tensor = inputs.get("labels_ner1")
            detached_labels_ner1 = labels_ner1_tensor.detach() if isinstance(labels_ner1_tensor, torch.Tensor) else labels_ner1_tensor

            labels_ner2_tensor = inputs.get("labels_ner2")
            detached_labels_ner2 = labels_ner2_tensor.detach() if isinstance(labels_ner2_tensor, torch.Tensor) else labels_ner2_tensor
            
            labels_tuple = (detached_labels_cls1, detached_labels_ner1, detached_labels_ner2)
            
            if loss is not None:
                loss = loss.detach()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits_tuple, labels_tuple)


class ModernBertCRFForTokenClassification(ModernBertPreTrainedModel):
    """
    NER（token classification）専用の ModernBERT + Linear + CRF.
    - Hugging Face の TokenClassifierOutput 互換に寄せ、`logits` は CRF への emissions を返す。
    - CRF decode / span化 / BIO制約チェック等は Trainer の外側で実施する想定。
    """
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.encoder = ModernBertModel(config)
        self.dropout = nn.Dropout(getattr(config, "classifier_dropout", 0.1))
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

        # BIO 制約付き初期化（label2id がある場合のみ）
        label2id = getattr(config, "label2id", None)
        if isinstance(label2id, dict) and "O" in label2id:
            with torch.no_grad():
                st, t, et = create_transitions(label2id)
                self.crf.start_transitions.copy_(st)
                self.crf.transitions.copy_(t)
                self.crf.end_transitions.copy_(et)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        """
        注意:
        - `sbintuitions/modernbert-ja-130m` のような base `ModernBertModel` のチェックポイントは
          state_dict のキーが `embeddings.*` のように prefix 無しで保存される。
        - 本クラスは encoder を `encoder.*` 配下に持つため、そのまま `super().from_pretrained()` だと
          encoder がロードされず「ほぼ全て新規初期化」になりやすい。
        そのため、ここでは encoder のみ `ModernBertModel.from_pretrained()` で確実にロードする。
        """
        import os
        # すでに本クラスの save_pretrained() 出力（trial_x等）であれば、通常の from_pretrained で全重みをロードする
        if isinstance(pretrained_model_name_or_path, str) and os.path.isdir(pretrained_model_name_or_path):
            has_weights = any(
                os.path.exists(os.path.join(pretrained_model_name_or_path, fn))
                for fn in ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json")
            )
            if has_weights:
                return super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **{k: v for k, v in kwargs.items() if k in ["cache_dir", "revision", "token"]})

        model = cls(config)

        # encoder を base モデルの重みでロード（Single-task用途の標準パス）
        encoder_kwargs = {k: v for k, v in kwargs.items() if k in ["cache_dir", "revision", "token", "local_files_only", "trust_remote_code"]}
        model.encoder = ModernBertModel.from_pretrained(pretrained_model_name_or_path, config=config, **encoder_kwargs)
        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[int] = None,  # Trainer から渡ってくることがあるが無視する
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        """
        labels:
          - shape (B, T)
          - ignore index は -100 を想定（HF流）
        """
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        sequence_output = enc_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # (B, T, C)

        loss = None
        if labels is not None:
            # CRF mask: attention_mask と labels != -100 のAND
            if attention_mask is None:
                mask = (labels != -100).bool()
            else:
                mask = attention_mask.bool() & (labels != -100).bool()

            # torchcrf の制約: mask の先頭 timestep は全サンプルで True 必須。
            # token classification では [CLS] を -100 にすることが多いので、その場合は先頭を落として CRF に渡す。
            emissions_for_crf = emissions
            labels_for_crf = labels
            mask_for_crf = mask
            if mask_for_crf.size(1) > 0 and not torch.all(mask_for_crf[:, 0]):
                emissions_for_crf = emissions_for_crf[:, 1:, :]
                labels_for_crf = labels_for_crf[:, 1:]
                mask_for_crf = mask_for_crf[:, 1:]

            # それでも timestep0 が False のサンプル（=有効トークンが無い/先頭が無効）を除外してCRF損失を計算する
            # torchcrf は batch 内で mask[:,0] が全Trueであることを要求するため。
            if mask_for_crf.size(1) == 0:
                # leaf tensor を返すと Trainer 側の in-place 演算で落ちるため、必ず計算グラフに接続した0にする
                loss = emissions.sum() * 0.0
            else:
                valid_batch = mask_for_crf[:, 0].bool()
                if torch.any(valid_batch):
                    ef = emissions_for_crf[valid_batch]
                    lf = labels_for_crf[valid_batch]
                    mf = mask_for_crf[valid_batch]
                    # mask=False 位置の labels はダミー値(0)にして範囲外アクセスを防ぐ
                    safe_labels = torch.where(mf, lf, torch.zeros_like(lf))
                    loss = -self.crf(
                        emissions=ef,
                        tags=safe_labels,
                        mask=mf,
                        reduction="mean",
                    )
                else:
                    loss = emissions.sum() * 0.0

        return TokenClassifierOutput(
            loss=loss,
            logits=emissions,
            hidden_states=enc_out.hidden_states,
            attentions=enc_out.attentions,
        )

    @torch.no_grad()
    def decode(
        self,
        emissions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> list[list[int]]:
        """
        CRF Viterbi decode（Trainer外で評価等に使用する想定）
        - attention_mask があればそれを優先
        - labels が与えられ、-100 が含まれる場合はそこもマスクする
        """
        if attention_mask is None:
            mask = None
        else:
            mask = attention_mask.bool()
        if labels is not None:
            label_mask = (labels != -100).bool()
            mask = label_mask if mask is None else (mask & label_mask)

        emissions_for_crf = emissions
        mask_for_crf = mask
        if mask_for_crf is not None and mask_for_crf.size(1) > 0 and not torch.all(mask_for_crf[:, 0]):
            emissions_for_crf = emissions_for_crf[:, 1:, :]
            mask_for_crf = mask_for_crf[:, 1:]

        return self.crf.decode(emissions_for_crf, mask=mask_for_crf)


class TokenClassificationDataCollator(DataCollatorWithPadding):
    """
    single-task token classification 用の軽量 collator.
    - features の `labels` または `labels_ner1` を取り込み、(B,T) に -100 パディングする。
    """
    def __init__(self, tokenizer, label_field: str = "labels", pad_to_multiple_of=None):
        super().__init__(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
        self.label_field = label_field

    def __call__(self, features):
        if not features:
            return {}

        # 入力 feature から labels を取り出す（labels_ner1 も受ける）
        labels_list = []
        for f in features:
            # 優先順位:
            # 1) 明示指定された label_field
            # 2) HF 標準の "labels"
            # 3) 互換: "labels_ner1"
            if self.label_field in f:
                labels_list.append(f.pop(self.label_field))
            elif "labels" in f:
                labels_list.append(f.pop("labels"))
            else:
                labels_list.append(f.pop("labels_ner1", None))

        batch = super().__call__(features)
        if "input_ids" not in batch or batch["input_ids"] is None:
            return batch

        device = batch["input_ids"].device
        seq_len = batch["input_ids"].size(1)

        padded = []
        for item in labels_list:
            if item is None:
                padded.append(torch.full((seq_len,), -100, dtype=torch.long, device=device))
                continue
            if isinstance(item, list):
                t = torch.tensor(item, dtype=torch.long, device=device)
            elif isinstance(item, torch.Tensor):
                t = item.to(dtype=torch.long, device=device)
            else:
                raise TypeError(f"labels item has unexpected type: {type(item)}")

            if t.numel() < seq_len:
                pad = torch.full((seq_len - t.numel(),), -100, dtype=torch.long, device=device)
                t = torch.cat([t, pad], dim=0)
            elif t.numel() > seq_len:
                t = t[:seq_len]
            padded.append(t)

        batch["labels"] = torch.stack(padded, dim=0)
        return batch