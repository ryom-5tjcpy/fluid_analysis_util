# 石原研究室用の流体解析のユーティリティ

このリポジトリは、石原研での研究活動において解析対象のデータを抽出するプログラムです。

## 実行方法

`main.py` は、データセット名と出力先をオプションで指定し、領域の原点・サイズ・取得列を位置引数で指定します。

```text
python main.py --dataset <データセット名> --output <出力ファイル> [--format csv|vtk] <x> <y> <z> <size> [<列名> ...]
```

- `--dataset`: 必須。`dataset_options.json` に定義されたデータセット名を指定します（例: `keta1_4096`、`keta2_4096`、`keta2_8192`）。
- `--output`: 必須。出力ファイルのパスを指定します。
- `--format`: 省略可能。`csv` または `vtk` を指定します。省略時は出力ファイルの拡張子（`.csv` または `.vtk`）から判定されます。指定した形式と拡張子が一致しない場合はエラーになります。
- `x`, `y`, `z`: 取得領域の原点座標を整数で指定します。
- `size`: 取得領域のサイズを整数で指定します。
- `列名`: 取得する列名を0個以上指定します。複数列は空白で区切ります。

CSV形式で `keta2_4096` データセットから取得する例:

```sh
python main.py --dataset keta2_4096 --output boxcell.csv \
  1 1 1 64 gnx gny nn u v w
```

VTK形式を明示して出力する例:

```sh
python main.py --dataset keta2_4096 --output boxcell.vtk --format vtk \
  1 1 1 64 gnx gny nn u v w
```
