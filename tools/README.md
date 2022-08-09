# Tools

## ttf2img.py

- help

    `python ttf2img.py --help`

    ```
    usage: ttf2img.py [-h] [--fft_file FFT_FILE] [--img_dir IMG_DIR]
                  [--mode {GB2312,GBK}]

    optional arguments:
    -h, --help           show this help message and exit
    --fft_file FFT_FILE  Path of ttf file
    --img_dir IMG_DIR    Path to save images
    --mode {GB2312,GBK}  Transfer mode
    ```

- run

    `python ttf2img.py --mode GB2312 --fft_file /path/to/fft/file --img_dir /path/to/save/transferred/images`

    ```
    --mode: GB2312 or GBK
    ```

## evaluate.py

- help

    `python evaluate.py --help`

    ```
    usage: evaluate.py [-h] [--fold1 FOLD1] [--fold2 FOLD2]

    optional arguments:
    -h, --help     show this help message and exit
    --fold1 FOLD1  Fold1
    --fold2 FOLD2  Fold2
    ```

- run

    `python evaluate.py --fold1 /path/to/image/fold --fold2 /path/to/image/fold`