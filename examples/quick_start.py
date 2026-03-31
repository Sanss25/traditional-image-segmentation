#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')

from segmentation import Segmenter

def create_test_image():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 60, (255, 255, 255), -1)
    cv2.rectangle(img, (50, 120), (150, 160), (200, 200, 200), -1)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None)
    args = parser.parse_args()
    
    print("🖼️  Traditional Image Segmentation Demo")
    print("=" * 50)
    
    segmenter = Segmenter()
    
    if args.image is None:
        print("📱 Using test image...")
        img = create_test_image()
    else:
        img = cv2.imread(args.image)
        if img is None:
            print("❌ Cannot load image, using test image")
            img = create_test_image()
    
    print("🔍 Running all methods...")
    results = segmenter.compare_all_methods(img)
    
    print("\n✅ SUCCESS! All methods work:")
    for method, mask in results.items():
        print(f"  {method:10}: {mask.sum()} pixels segmented")
    
    print("\n🎉 Run with real image: python examples/quick_start.py --image your_image.jpg")

if __name__ == "__main__":
    main()
