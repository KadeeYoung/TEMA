"""Free-running dialogue rollout with model-generated assistant history."""

from __future__ import annotations

import re

import sys

import time

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from tema_chat.evaluation.sidecar import AUDIO_LABEL_RE, load_sidecar

from tema_chat.rewards.parsing import answer_reward, consistency_component_reward, consistency_reward, format_reward as training_format_reward, route_reward as training_route_reward, span_reward as training_span_reward

from tema_chat.evaluation.engine import ANSWER_RE, dense_span_metrics, extract_yes_no, interval_set_iou, parse_format, parse_route_set, parse_spans, route_set_f1

SIMPLE_EVIDENCE_RE = re.compile(r'^\s*Evidence\s+audios?\s*:\s*(.*?)\s*$', re.I | re.M)


SIMPLE_ANSWER_RE = re.compile(r'^\s*Final\s+answer\s*:\s*(.*?)\s*$', re.I | re.M)


ORACLE_ROUTE_SPAN_RE = re.compile(
    r'^\s*<think>\s*(<route>.*?</route>)\s*(<span>.*?</span>)\s*<reason>',
    re.S,
)


SIMPLE_TURN_REMINDER = (
    'Respond now with exactly two lines headed "Evidence audios:" and "Final answer:". '
    'On the first line, list every required Clip label.'
)


ORACLE_EVIDENCE_ANSWER_PROMPT_TEMPLATE = (
    'Evidence for the current question:\n'
    '{route_block}\n'
    '{span_block}\n\n'
    'Reply with exactly one lines:\n'
    'Final answer: ...'
)


EXPLAINED_ORACLE_EVIDENCE_ANSWER_PROMPT_TEMPLATE = (
    'Interpret the evidence annotations as follows:\n'
    '<route> lists all audio clips that must be inspected to answer the current question.\n'
    '<span> lists every occurrence of the event or events needed to answer the current question in each routed '
    'audio as start-end time intervals in seconds.\n'
    'Audios{k}[NONE] means Audio k was inspected but has no matching occurrence.\n'
    'An audio absent from <route> is not needed for the current question.\n\n'
    'Evidence for the current question:\n'
    '{route_block}\n'
    '{span_block}\n\n'
    'Reply with exactly one lines:\n'
    'Final answer: ...'
)


SIMPLE_SYSTEM_PROMPT = """You are answering a continuing conversation about audio clips. Clips are numbered in upload order, and each newly uploaded audio is explicitly labeled in the user message.

Reply with exactly two non-empty lines and no other text:
Evidence audios: [actual required clip labels]
Final answer: [concise direct answer]

Replace both bracketed instructions with actual content and never repeat the brackets. On the Evidence audios line, write the word Clip before each actual numeric label. The Evidence audios field is a routing decision. Include every clip that must be inspected to answer the current question, including every clip involved in a comparison or joint inference. Do not list only the winning clip named in the final answer. Do not list clips that are merely present in the conversation but are unnecessary for the current question. Do not output XML tags, hidden reasoning, or additional fields."""


def build_oracle_route_prefix(gold_text: str) -> str:
    match = ORACLE_ROUTE_SPAN_RE.match(gold_text)
    if match is None:
        raise ValueError('gold response does not begin with a route/span/reason block')
    route_block = match.group(1).strip()
    return f'<think>\n{route_block}\n'


def build_oracle_route_span_prefix(gold_text: str) -> str:
    match = ORACLE_ROUTE_SPAN_RE.match(gold_text)
    if match is None:
        raise ValueError('gold response does not begin with a route/span/reason block')
    route_block, span_block = (part.strip() for part in match.groups())
    return f'<think>\n{route_block}\n{span_block}\n<reason>'


def build_oracle_evidence_answer_prompt(gold_text: str, explain_schema: bool = False) -> str:
    match = ORACLE_ROUTE_SPAN_RE.match(gold_text)
    if match is None:
        raise ValueError('gold response does not begin with a route/span/reason block')
    route_block, span_block = (part.strip() for part in match.groups())
    template = (
        EXPLAINED_ORACLE_EVIDENCE_ANSWER_PROMPT_TEMPLATE
        if explain_schema
        else ORACLE_EVIDENCE_ANSWER_PROMPT_TEMPLATE
    )
    return (
        template
        .replace('{route_block}', route_block)
        .replace('{span_block}', span_block)
    )


def build_answer_scoring_completion(gold_text: str, answer: str) -> str:
    prefix = build_oracle_route_span_prefix(gold_text)
    return (
        f'{prefix}The route and span were supplied.</reason>\n'
        f'</think>\n<answer>{answer}</answer>'
    )


def normalize_answer_only_for_scoring(answer: str, task_type: str) -> str:
    value = answer.strip()
    if task_type == 'A3':
        if re.fullmatch(r'\d+[.!]?', value):
            return f'{value.rstrip(".!")} times'
    if task_type in {'A5', 'A5-gap', 'A17', 'A18'}:
        if re.fullmatch(r'-?\d+(?:\.\d+)?[.!]?', value):
            return f'{value.rstrip(".!")} seconds'
    if task_type in {'A7', 'A8', 'A9', 'A10', 'A13'}:
        if (
            not re.search(r'\b(?:clips?|audios?)\b', value, re.I)
            and re.fullmatch(r'\d+(?:\s*(?:,|and|&)\s*\d+)*[.!]?', value, re.I)
        ):
            return f'Clips {value}'
    return value


def parse_simple_response(text: str) -> Tuple[Dict[str, bool], Optional[Set[int]], Optional[str]]:
    evidence_matches = list(SIMPLE_EVIDENCE_RE.finditer(text))
    answer_match = SIMPLE_ANSWER_RE.search(text)
    route: Optional[Set[int]] = None
    evidence_parse = False
    parsed_routes: List[Set[int]] = []
    for evidence_match in evidence_matches:
        body = evidence_match.group(1).strip()
        if (
            '<' in body
            or '[' in body
            or re.search(r'\b(?:example|required clip|must be checked|actual)\b', body, re.I)
        ):
            parsed_routes = []
            break
        indices = {int(value) for value in re.findall(r'\d+', body)}
        if indices:
            parsed_routes.append(indices)
        elif re.fullmatch(r'(?:none|n/?a|not applicable)', body, re.I):
            parsed_routes.append(set())
        else:
            parsed_routes = []
            break
    if parsed_routes and not (any(parsed_routes) and any(not part for part in parsed_routes)):
        route = set().union(*parsed_routes)
        evidence_parse = True

    final_answer = answer_match.group(1).strip() if answer_match else None
    answer_parse = bool(final_answer)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    exactly_two_fields = bool(
        len(lines) == 2
        and SIMPLE_EVIDENCE_RE.fullmatch(lines[0])
        and SIMPLE_ANSWER_RE.fullmatch(lines[1])
    )
    has_structured_tags = any(tag in text.lower() for tag in ('<route>', '<span>', '<reason>', '<answer>', '<think>'))
    pred_format = {
        'format_ok': exactly_two_fields and evidence_parse and answer_parse and not has_structured_tags,
        'route_parse': evidence_parse,
        'span_parse': False,
        'reason_parse': False,
        'answer_parse': answer_parse,
        'empty': len(text.strip()) == 0,
    }
    return pred_format, route, final_answer


def parse_answer_only_response(text: str) -> Tuple[Dict[str, bool], Optional[str]]:
    answer_match = SIMPLE_ANSWER_RE.search(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    has_structured_tags = any(
        tag in text.lower()
        for tag in ('<route>', '<span>', '<reason>', '<answer>', '<think>')
    )
    if answer_match:
        final_answer = answer_match.group(1).strip()
    elif text.strip() and not has_structured_tags:
        final_answer = text.strip()
    else:
        final_answer = None
    answer_parse = bool(final_answer)
    exactly_one_field = bool(
        len(lines) == 1
        and SIMPLE_ANSWER_RE.fullmatch(lines[0])
    )
    pred_format = {
        'format_ok': exactly_one_field and answer_parse and not has_structured_tags,
        'route_parse': False,
        'span_parse': False,
        'reason_parse': False,
        'answer_parse': answer_parse,
        'empty': len(text.strip()) == 0,
    }
    return pred_format, final_answer


def extract_plain_yes_no(answer: Optional[str]) -> Optional[bool]:
    if not answer:
        return None
    prefix = re.split(r'[\s.,!:;]', answer.strip().lower(), maxsplit=1)[0]
    if prefix in ('yes', 'yeah', 'yep'):
        return True
    if prefix in ('no', 'nope'):
        return False
    return None


def label_audio_placeholders(content: str, first_clip_index: int) -> str:
    next_index = first_clip_index

    def replace(_match: re.Match) -> str:
        nonlocal next_index
        label = f'<audio> This newly uploaded audio is Clip {next_index}.'
        next_index += 1
        return label

    return re.sub(r'<audio>', replace, content)


def sidecar_route_indices(sidecar_row: Mapping[str, Any]) -> Set[int]:
    clip_to_index: Dict[str, int] = {}
    for label, audio in (sidecar_row.get('audios') or {}).items():
        labels = AUDIO_LABEL_RE.findall(str(label))
        if len(labels) != 1:
            raise ValueError(f'cannot parse audio label {label!r}')
        clip = audio.get('clip')
        if clip is not None:
            clip_to_index[str(clip)] = int(labels[0])

    result: Set[int] = set()
    for clip in sidecar_row.get('route') or []:
        clip = str(clip)
        if clip not in clip_to_index:
            raise ValueError(f'route clip {clip!r} is absent from sidecar audio mapping')
        result.add(clip_to_index[clip])
    return result


def build_turn_specs(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    sidecar_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    pending_user: Optional[Dict[str, str]] = None
    audio_count = 0
    for message in row['messages']:
        role = message.get('role')
        if role == 'user':
            if pending_user is not None:
                raise ValueError('two user messages without an assistant response')
            pending_user = {'role': 'user', 'content': str(message.get('content', ''))}
            audio_count += pending_user['content'].count('<audio>')
        elif role == 'assistant':
            if pending_user is None:
                raise ValueError('assistant message without a preceding user message')
            turn_index = len(specs)
            sidecar_row = sidecar_rows[turn_index]
            gold_route = sidecar_route_indices(sidecar_row)
            text_route = parse_route_set(str(message.get('content', '')))
            if text_route != gold_route:
                raise ValueError(
                    f'gold route mismatch at turn {turn_index}: text={text_route}, sidecar={gold_route}'
                )
            specs.append({
                'turn_index': turn_index,
                'user_message': pending_user,
                'gold_text': str(message.get('content', '')),
                'audio_count': audio_count,
                'task_type': sidecar_row.get('task_type'),
                'answer_struct': sidecar_row.get('answer_struct'),
                'gold_route': gold_route,
                'audio_lengths': {
                    int(match.group(1)): float(metadata['audio_len'])
                    for label, metadata in (sidecar_row.get('audios') or {}).items()
                    if (match := AUDIO_LABEL_RE.fullmatch(str(label)))
                    and isinstance(metadata, Mapping)
                    and metadata.get('audio_len') is not None
                },
            })
            pending_user = None
        else:
            raise ValueError(f'unsupported role in processed dialogue: {role!r}')
    if pending_user is not None:
        raise ValueError('dialogue ends with an unanswered user message')
    if len(specs) != len(sidecar_rows):
        raise ValueError(f'turn/sidecar mismatch: {len(specs)} != {len(sidecar_rows)}')
    if audio_count != len(row['audios']):
        raise ValueError(f'audio token/path mismatch: {audio_count} != {len(row["audios"])}')
    return specs


def score_prediction(item: Mapping[str, Any], pred_text: str, response_style: str) -> Dict[str, Any]:
    gold_text = item['gold_text']
    gold_route = set(item['gold_route'])
    gold_spans = parse_spans(gold_text) or {}
    if response_style == 'structured':
        pred_format = parse_format(pred_text)
        pred_route = parse_route_set(pred_text)
        pred_spans = parse_spans(pred_text)
        answer_match = ANSWER_RE.search(pred_text)
        pred_final_answer = answer_match.group(1).strip() if answer_match else None
        pred_exists = extract_yes_no(pred_text)
    elif response_style == 'simple':
        pred_format, pred_route, pred_final_answer = parse_simple_response(pred_text)
        pred_spans = None
        pred_exists = extract_plain_yes_no(pred_final_answer)
    else:
        pred_format, pred_final_answer = parse_answer_only_response(pred_text)
        pred_route = None
        pred_spans = None
        pred_exists = extract_plain_yes_no(pred_final_answer)

    record: Dict[str, Any] = {
        'dialogue_idx': item['dialogue_idx'],
        'source_dialogue_idx': item['source_dialogue_idx'],
        'turn_index': item['turn_index'],
        'group_id': item['group_id'],
        'ticket_id': item['ticket_id'],
        'task_type': item['task_type'],
        'answer_struct': item.get('answer_struct'),
        'audio_lengths': item.get('audio_lengths') or {},
        'response_style': response_style,
        'history_mode': 'free_running',
        'structured_turn_prompt': item.get('structured_turn_prompt'),
        'oracle_evidence_prompt': item.get('oracle_evidence_prompt'),
        'oracle_evidence_schema_explained': item.get('oracle_evidence_schema_explained', False),
        'prior_model_turns': item['turn_index'],
        'audio_count': item['audio_count'],
        'gold_evidence_audios': sorted(gold_route),
        'pred_evidence_audios': sorted(pred_route) if pred_route is not None else None,
        'gold_text': gold_text,
        'pred_text': pred_text,
        'pred_final_answer': pred_final_answer,
        'gold_format': parse_format(gold_text),
        'pred_format': pred_format,
    }

    if pred_route is not None:
        precision, recall, f1 = route_set_f1(gold_route, pred_route)
        record.update({
            'route_precision': precision,
            'route_recall': recall,
            'route_set_f1': f1,
            'route_exact': pred_route == gold_route,
        })

    if response_style == 'structured' and pred_spans is not None and gold_spans:
        ious = []
        for label, gold_intervals in gold_spans.items():
            pred_intervals = pred_spans.get(label)
            ious.append(0.0 if pred_intervals is None else interval_set_iou(gold_intervals, pred_intervals))
        if ious:
            record['span_iou'] = sum(ious) / len(ious)
    if response_style == 'structured':
        record.update(dense_span_metrics(gold_spans, pred_spans, gold_route, pred_route))

    answer_struct = item.get('answer_struct')
    if response_style in {'structured', 'answer_only'}:
        task_type = str(item.get('task_type') or 'unknown')
        answer_completion = pred_text
        if response_style == 'answer_only' and pred_final_answer:
            strict_answer_completion = build_answer_scoring_completion(gold_text, pred_final_answer)
            strict_answer_score = answer_reward(
                strict_answer_completion,
                task_type,
                answer_struct or {},
                gold_text,
                item.get('audio_lengths'),
                allow_exact_match=False,
            )
            normalized_answer = normalize_answer_only_for_scoring(pred_final_answer, task_type)
            answer_completion = build_answer_scoring_completion(gold_text, normalized_answer)
            record['answer_score_strict'] = strict_answer_score
            record['answer_exact_strict'] = strict_answer_score >= 1.0 - 1e-9
            record['answer_scoring_text'] = normalized_answer
            record['answer_scoring_normalized'] = normalized_answer != pred_final_answer
        gold_answer_score = answer_reward(
            gold_text,
            task_type,
            answer_struct or {},
            gold_text,
            item.get('audio_lengths'),
            allow_exact_match=False,
        )
        record['answer_gold_valid'] = gold_answer_score >= 1.0 - 1e-9
        semantic_answer_score = answer_reward(
            answer_completion,
            task_type,
            answer_struct or {},
            gold_text,
            item.get('audio_lengths'),
            allow_exact_match=False,
        )
        if record['answer_gold_valid']:
            record['answer_score'] = semantic_answer_score
            record['answer_exact'] = semantic_answer_score >= 1.0 - 1e-9
        consistency_score, predicted_consistency_applicable = consistency_reward(
            answer_completion,
            task_type,
            answer_struct or {},
            audio_lengths=item.get('audio_lengths'),
        )
        gold_consistency_score, gold_consistency_applicable = consistency_reward(
            gold_text,
            task_type,
            answer_struct or {},
            audio_lengths=item.get('audio_lengths'),
        )
        record['answer_consistency_applicable'] = gold_consistency_applicable
        record['answer_consistency_gold_valid'] = (
            not gold_consistency_applicable or gold_consistency_score >= 1.0 - 1e-9
        )
        if gold_consistency_applicable and record['answer_consistency_gold_valid']:
            record['answer_consistency_score'] = (
                consistency_score if predicted_consistency_applicable else 0.0
            )
        if (
            response_style == 'structured'
            and record['answer_gold_valid']
            and record['answer_consistency_gold_valid']
        ):
            reward_components = {
                'format': training_format_reward(pred_text, item.get('audio_lengths')),
                'route': training_route_reward(
                    pred_text, sorted(gold_route), item.get('audio_lengths')
                ),
                'span': training_span_reward(
                    pred_text,
                    sorted(gold_route),
                    gold_spans,
                    item.get('audio_lengths'),
                ),
                'answer': semantic_answer_score,
                'consistency': consistency_component_reward(
                    pred_text,
                    task_type,
                    answer_struct or {},
                    audio_lengths=item.get('audio_lengths'),
                ),
            }
            record['reward_components'] = reward_components
            record['reward_score'] = (
                0.10 * reward_components['format']
                + 0.05 * reward_components['route']
                + 0.40 * reward_components['span']
                + 0.40 * reward_components['answer']
                + 0.15 * reward_components['consistency']
            )
    if answer_struct and answer_struct.get('type') == 'existence' and isinstance(answer_struct.get('exists'), bool):
        record['gold_exists'] = answer_struct['exists']
        record['pred_exists'] = pred_exists
        if pred_exists is not None:
            record['existence_correct'] = pred_exists == answer_struct['exists']
    return record


def run_rollout(
    engine: Any,
    indexed: Sequence[Tuple[int, Dict[str, Any], Dict[str, Any]]],
    sidecar_by_key: Mapping[Tuple[Optional[str], Optional[str]], List[Dict[str, Any]]],
    response_style: str,
    max_new_tokens: int,
    oracle_route: bool = False,
    oracle_route_span: bool = False,
    structured_turn_prompt: Optional[str] = None,
    explain_oracle_evidence: bool = False,
    request_types: Optional[Tuple[Any, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    if request_types is None:
        from swift.infer_engine import InferRequest, RequestConfig
    else:
        # Native-HF engines reuse the same history and scoring protocol.
        InferRequest, RequestConfig = request_types

    states = []
    for dialogue_idx, row, manifest in indexed:
        key = (manifest.get('group_id'), manifest.get('ticket_id'))
        specs = build_turn_specs(row, manifest, sidecar_by_key[key])
        history: List[Dict[str, str]] = []
        if response_style == 'simple':
            history.append({'role': 'system', 'content': SIMPLE_SYSTEM_PROMPT})
        states.append({
            'dialogue_idx': dialogue_idx,
            'source_dialogue_idx': manifest['source_dialogue_idx'],
            'group_id': key[0],
            'ticket_id': key[1],
            'row': row,
            'specs': specs,
            'history': history,
        })

    records: List[Dict[str, Any]] = []
    round_seconds: List[float] = []
    max_turns = max((len(state['specs']) for state in states), default=0)
    config = RequestConfig(max_tokens=max_new_tokens, temperature=0.0)
    for turn_index in range(max_turns):
        active = [state for state in states if turn_index < len(state['specs'])]
        requests = []
        items = []
        for state in active:
            spec = state['specs'][turn_index]
            response_prefix = None
            oracle_evidence_prompt = None
            user_message = dict(spec['user_message'])
            if response_style in {'simple', 'answer_only'}:
                new_audio_count = user_message['content'].count('<audio>')
                first_clip_index = spec['audio_count'] - new_audio_count + 1
                user_message['content'] = label_audio_placeholders(
                    user_message['content'],
                    first_clip_index,
                )
                if response_style == 'simple':
                    user_message['content'] += f'\n\n{SIMPLE_TURN_REMINDER}'
                else:
                    oracle_evidence_prompt = build_oracle_evidence_answer_prompt(
                        spec['gold_text'],
                        explain_schema=explain_oracle_evidence,
                    )
                    user_message['content'] += f'\n\n{oracle_evidence_prompt}'
            elif structured_turn_prompt:
                user_message['content'] += f'\n\n{structured_turn_prompt}'
            state['history'].append(user_message)
            request_kwargs = {}
            if oracle_route_span:
                response_prefix = build_oracle_route_span_prefix(spec['gold_text'])
                request_kwargs['chat_template_kwargs'] = {'response_prefix': response_prefix}
            elif oracle_route:
                response_prefix = build_oracle_route_prefix(spec['gold_text'])
                request_kwargs['chat_template_kwargs'] = {'response_prefix': response_prefix}
            requests.append(
                InferRequest(
                    messages=[dict(message) for message in state['history']],
                    audios=list(state['row']['audios'][: spec['audio_count']]),
                    **request_kwargs,
                )
            )
            items.append({
                **spec,
                'dialogue_idx': state['dialogue_idx'],
                'source_dialogue_idx': state['source_dialogue_idx'],
                'group_id': state['group_id'],
                'ticket_id': state['ticket_id'],
                'oracle_response_prefix': response_prefix,
                'structured_turn_prompt': structured_turn_prompt,
                'oracle_evidence_prompt': oracle_evidence_prompt,
                'oracle_evidence_schema_explained': (
                    explain_oracle_evidence if response_style == 'answer_only' else False
                ),
            })

        started = time.time()
        responses = engine.infer(requests, config)
        elapsed = time.time() - started
        round_seconds.append(elapsed)
        print(
            f'[rollout] turn_index={turn_index} dialogues={len(active)} generation_seconds={elapsed:.1f}',
            file=sys.stderr,
        )
        for state, item, response in zip(active, items, responses):
            choice = response.choices[0]
            response_text = choice.message.content or ''
            response_prefix = item['oracle_response_prefix']
            if response_prefix is not None and response_text.startswith(response_prefix):
                pred_text = response_text
                generated_continuation = response_text[len(response_prefix):]
            elif response_prefix is not None:
                pred_text = response_prefix + response_text
                generated_continuation = response_text
            else:
                pred_text = response_text
                generated_continuation = response_text
            record = score_prediction(item, pred_text, response_style)
            record['oracle_route'] = oracle_route
            record['oracle_route_span'] = oracle_route_span
            record['oracle_response_prefix'] = response_prefix
            record['generated_continuation'] = generated_continuation
            record['finish_reason'] = getattr(choice, 'finish_reason', None)
            usage = getattr(response, 'usage', None)
            record['completion_tokens'] = getattr(usage, 'completion_tokens', None)
            for native_field in ['native_generated_token_ids', 'native_decoded_text']:
                if hasattr(choice, native_field):
                    record[native_field] = getattr(choice, native_field)
            records.append(record)
            state['history'].append({'role': 'assistant', 'content': getattr(choice, 'history_content', pred_text)})
    records.sort(key=lambda record: (record['dialogue_idx'], record['turn_index']))
    return records, round_seconds

