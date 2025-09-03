# aggregate.py
import argparse, json, glob
import numpy as np
from pathlib import Path
from collections import defaultdict

def key_of(rec):
    # loss까지 포함해 조합을 유일화
    return (
        rec["src"], rec["tgt"],
        float(rec["src_prior"]), float(rec["tgt_prior"]),
        rec["arch"], rec.get("loss", "unknown")
    )

def fmt_mean_std(arr):
    arr = np.array(arr, dtype=float)
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    return f"{mu:.4f} ({sd:.3f})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, default=Path("./metric/lin"),
                    help="evaluate.py가 만든 결과 폴더 (예: ./metric/lin)")
    ap.add_argument("--out_txt", type=Path, default=None,
                    help="요약 txt 경로 (기본: in_dir/summary_mean_std_by_loss.txt)")
    args = ap.parse_args()

    if args.out_txt is None:
        args.out_txt = args.in_dir / "summary_mean_std_by_loss4.txt"

    # 폴더 계층 안의 results.jsonl 를 전부 찾는다 (과거 패턴도 겸용하고 싶으면 + results_*.jsonl 추가)
    files = sorted(set(
        glob.glob(str(args.in_dir / "**/results.jsonl"), recursive=True)
        # + glob.glob(str(args.in_dir / "**/results_*.jsonl"), recursive=True)  # 필요시 활성화
    ))

    if not files:
        print(f"[aggregate] no result files found in {args.in_dir}")
        return

    buckets = defaultdict(list)

    # 여러 줄일 수도 있으니 안전하게 모두 읽기
    for f in files:
        with open(f, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                buckets[key_of(rec)].append(rec)

    lines = []
    lines.append("=== Evaluation Summary (mean ± std across seeds, grouped by loss_type) ===")
    lines.append("PAIR\tPRIORS\tARCH\tLOSS\tN\tACC\tMacroF1\tAUC\tTestLoss")

    rows = []
    for k, items in sorted(buckets.items()):
        src, tgt, sp, tp, arch, loss = k
        N = len(items)
        accs = [it.get("test_acc", np.nan) for it in items]
        f1s  = [it.get("test_macro_f1", np.nan) for it in items]
        aucs = [it.get("test_auc", np.nan) for it in items]
        tloss= [it.get("test_loss", np.nan) for it in items]

        rows.append([
            f"{src}->{tgt}",
            f"({sp:.1f},{tp:.1f})",
            arch,
            loss,
            str(N),
            fmt_mean_std(accs),
            fmt_mean_std(f1s),
            fmt_mean_std(aucs),
            fmt_mean_std(tloss),
        ])

    # 보기 좋게 정렬: pair, priors, arch, loss
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

    for r in rows:
        lines.append("\t".join(r))

    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[aggregate] saved: {args.out_txt}")

if __name__ == "__main__":
    main()
