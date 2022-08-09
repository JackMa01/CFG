import os
import pygame
import argparse

pygame.init()


parser = argparse.ArgumentParser()
parser.add_argument('--fft_file', type=str, default='', help='Path of ttf file')
parser.add_argument('--img_dir', type=str, default='', help='Path to save images')
parser.add_argument('--mode', type=str, default='GB2312', choices=['GB2312', 'GBK'], help='Transfer mode')
args = parser.parse_args()

if not os.path.exists(args.img_dir):
    os.makedirs(args.img_dir)


fonts = []

for zone in range(72):
    zone_data = 0xB0A0 + zone * 0x0100

    for pos in range(6*16):

        if pos in [0, 6*16-1]:
            continue
        pos_data = zone_data + pos

        if pos_data in [0xD7FA, 0xD7FB, 0xD7FC, 0xD7FD, 0xD7FE]:
            continue
        pos_data = pos_data.to_bytes(2, byteorder='big')

        font = pos_data.decode('gbk')
        fonts.append(font)


if args.mode == 'GBK':

    for zone in range(32):
        zone_data = 0x8140 + zone * 0x0100

        for pos in range(12*16):
            
            if pos in [3*16+15, 11*16+15]:
                continue
            pos_data = zone_data + pos
            pos_data = pos_data.to_bytes(2, byteorder='big')
            
            font = pos_data.decode('gbk')
            fonts.append(font)


    for zone in range(85):
        zone_data = 0xAA40 + zone * 0x0100

        for pos in range(6*16+1):
            
            if pos in [3*16+15]:
                continue
            pos_data = zone_data + pos

            if pos_data >= 0xFE50:
                continue
            pos_data = pos_data.to_bytes(2, byteorder='big')
            
            font = pos_data.decode('gbk')
            fonts.append(font)


print('Fonts number: {}'.format(len(fonts)))


crop_size = 256
font_render = pygame.font.Font(args.fft_file, crop_size)

for i, character in enumerate(fonts):
    character_render = font_render.render(character, True, (0,0,0), (255,255,255))
    character_img = pygame.transform.scale(character_render, (crop_size, crop_size))
    pygame.image.save(character_img, os.path.join(args.img_dir, '{}.png'.format(i)))

    print('\rRenderring: {} / {}'.format(i+1, len(fonts)), end='')

print()
