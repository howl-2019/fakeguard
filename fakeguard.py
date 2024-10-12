import sys
import argparse
from torchvision import transforms
from opt import PoisonGeneration

def crop_to_square(img):
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ]
    )
    return image_transforms(img)


def main():
    poison_generator = PoisonGeneration(device="cuda", eps=args.eps)
    #all_data_paths = glob.glob(os.path.join(args.directory, "*.p"))

    # 이미지를 열기 전에 경로를 사용합니다.
    src_image_path = args.source
    #all_texts = [pickle.load(open(f, "rb"))['text'] for f in all_data_paths]
    target_image_path = args.target

    # 경로 문자열을 사용하여 generate_one 메서드 호출
    poison_generator.generate_one(src_image_path, target_image_path)
    #os.makedirs(args.outdir, exist_ok=True)

    # all_result_imgs는 이미지 객체 하나를 반환하므로 리스트로 변환
    #cur_data = {"text": all_texts[0], "img": all_result_imgs}  # 첫 번째 텍스트와 이미지 저장
    #pickle.dump(cur_data, open(os.path.join(args.outdir, "0.p"), "wb"))

    print("Image processing completed!")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    #parser.add_argument('-d', '--directory', type=str, help="", default='')
    #parser.add_argument('-od', '--outdir', type=str, help="", default='')
    parser.add_argument('-e', '--eps', type=float, default=0.04)
    parser.add_argument('-s', '--source', type=str, default='image/bboriginal.png')  # 경로 문자열
    parser.add_argument('-t', '--target', type=str, default='image/bbanimation.png')  # 경로 문자열
    return parser.parse_args(argv)

if __name__ == '__main__':
    import time

    args = parse_arguments(sys.argv[1:])
    main()