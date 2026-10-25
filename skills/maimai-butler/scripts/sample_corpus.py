"""从微博语料库里按月加权抽样本，输出到单独文件（该文件已被 .gitignore 排除）。

为什么要按月加权，而不是全库随机抽
    语料是按时间倒序抓的，越新的月份越密。全库直接随机抽 120 条，
    2026-10（当月未完，只抓到 25 条）和 2025-11（抓到 11 条）这种
    存量小的月份很可能一条都抽不到，样本就会丢掉一年里最早和最新的声线。
    按月分层、按各月存量比例分配名额，抽出来的分布跟全库一致，
    既不会让小月份被挤掉，也不会让小月份被抽干。

名额怎么分（最大余数法）
    1. 每个月的理论名额 = 该月存量 / 全库总量 * N
    2. 先取整数部分
    3. 剩下的名额按小数部分从大到小补，且不超过该月存量
    这样 12 个月的名额加起来正好等于 N，不会出现四舍五入导致的总数偏差。

用法
    python scripts/sample_corpus.py                        # 默认 120 条，每次结果不同
    python scripts/sample_corpus.py --n 60                 # 抽 60 条
    python scripts/sample_corpus.py --seed 7               # 手动指定种子（复现用）
    python scripts/sample_corpus.py --show-alloc           # 只看名额分配不落盘

种子策略
    默认不给种子，用**当前时间**生成，所以每次跑抽出来的样本都不一样。
    运行时会把这个时间种子打印出来——想原样复现某一次的结果，
    把打印出的数字传给 --seed 即可。

输出
    corpus/weibo-sample.jsonl，字段与语料库一致（mid / date / text），
    按月升序排列，同月内随机。该文件在 .gitignore 里，不进公开仓库。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# 脚本位于本 skill 的 scripts/ 下，语料在同级 corpus/。
# 用相对 __file__ 定位，整个 skill 目录拷到哪儿都能跑（不依赖当前工作目录）。
SKILL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = SKILL_ROOT / "corpus" / "weibo-corpus.jsonl"
DEFAULT_OUT = SKILL_ROOT / "corpus" / "weibo-sample.jsonl"

def resolve_seed(explicit: int | None) -> int:
    """确定这次用的随机种子。

    没显式传就用当前时间（毫秒）当种子，因此每次跑结果都不同。
    用毫秒而不是纳秒：数字短、好抄，而一次完整运行远不止 1 毫秒，
    连续两次跑不可能撞到同一个值。
    """
    if explicit is not None:
        return explicit
    return int(time.time() * 1000)


def load_jsonl(path: Path) -> list[dict]:
    """读 jsonl。空行跳过，坏行报出文件名和行号后退出——静默丢数据比报错更糟。"""
    if not path.exists():
        sys.exit(f"找不到语料文件：{path}")
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            sys.exit(f"{path.name} 第 {lineno} 行不是合法 JSON：{exc}")
    return rows


def month_key(rec: dict) -> str:
    """取年月作为分层键。date 为空或格式不对时归入 'unknown'，不丢弃。"""
    day = (rec.get("date") or "").strip()
    return day[:7] if len(day) >= 7 else "unknown"


def allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """按各层存量比例分配 n 个名额，用最大余数法凑满。

    counts: {月份: 该月存量}
    返回:   {月份: 分配名额}，总和恰好为 n，且每层名额不超过其存量。
    """
    total = sum(counts.values())
    if n > total:
        sys.exit(f"要抽 {n} 条，但语料库只有 {total} 条")

    raw = {m: counts[m] / total * n for m in counts}
    alloc = {m: int(raw[m]) for m in counts}          # 先取整数部分

    # 余数从大到小排队，把剩下的名额补出去
    left = n - sum(alloc.values())
    # 平手时用月份字符串兜底排序，保证结果确定（同一 seed 必得同一样本）
    order = sorted(counts, key=lambda m: (-(raw[m] - alloc[m]), m))
    for m in order:
        if left <= 0:
            break
        # 名额不能超过该月存量，否则 random.sample 会直接抛错
        if alloc[m] < counts[m]:
            alloc[m] += 1
            left -= 1

    # 兜底：极端情况下（层数很少、存量极不均衡）上面可能补不满，
    # 再按存量从大到小扫一遍，只要还有余量就继续补。
    if left > 0:
        for m in sorted(counts, key=lambda x: (-counts[x], x)):
            while left > 0 and alloc[m] < counts[m]:
                alloc[m] += 1
                left -= 1
            if left == 0:
                break

    return alloc


def main() -> int:
    ap = argparse.ArgumentParser(description="按月加权抽取微博语料样本")
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=120, help="抽样条数，默认 120")
    ap.add_argument("--seed", type=int, default=None,
                    help="随机种子。不传则用当前时间，每次结果不同；传了就复现那一次")
    ap.add_argument("--show-alloc", action="store_true", help="只打印名额分配，不写文件")
    args = ap.parse_args()

    # 到这里才定种子：不传就是用时间，传了就用传的
    seed = resolve_seed(args.seed)

    rows = load_jsonl(args.input)
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in rows:
        groups[month_key(rec)].append(rec)

    counts = {m: len(v) for m, v in groups.items()}
    alloc = allocate(counts, args.n)

    # 每层独立用 seed 派生一个随机源：某一月增删数据不会连带改变其他月的抽样结果
    picked: list[dict] = []
    for month in sorted(groups):
        k = alloc.get(month, 0)
        if k <= 0:
            continue
        rng = random.Random(f"{seed}:{month}")
        picked.extend(rng.sample(groups[month], k))

    # 输出按月升序；同月内再按日期倒序（新→旧），保持和语料库的阅读习惯一致
    picked.sort(key=lambda r: (r.get("date") or "", ), reverse=True)
    picked.sort(key=lambda r: (r.get("date") or "")[:7])

    print(f"读入 {len(rows)} 条，{len(groups)} 个月份")
    print(f"目标 {args.n} 条，实抽 {len(picked)} 条")
    if args.seed is None:
        print(f"种子 {seed}（时间种子，每次结果不同）")
        print(f"     要原样复现这次：--seed {seed}")
    else:
        print(f"种子 {seed}（手动指定）")
    print()
    print(f"{'月份':<10}{'存量':>6}{'名额':>6}{'抽样占比':>10}")
    print("-" * 34)
    stat: Counter[str] = Counter()
    for rec in picked:
        stat[month_key(rec)] += 1
    for month in sorted(counts):
        k = stat.get(month, 0)
        pct = f"{k / counts[month] * 100:.0f}%" if counts[month] else "-"
        print(f"{month:<10}{counts[month]:>6}{k:>6}{pct:>10}")
    print("-" * 34)
    print(f"{'合计':<10}{len(rows):>6}{len(picked):>6}")

    if args.show_alloc:
        print("\n--show-alloc：不写文件")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for rec in picked:
            # 只写约定的三个字段，不把语料库的其他字段带出去
            row = {"mid": rec.get("mid", ""), "date": rec.get("date", ""),
                   "text": rec.get("text", "")}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
