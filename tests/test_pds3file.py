# -*- coding: utf-8 -*-
import pytest
import os
import numpy
import gzip
import bz2
from numpy.testing import assert_almost_equal
from planetaryimage import PDS3Image
from pvl._collections import Units
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/')
filename = os.path.join(DATA_DIR, 'pds3_1band.IMG')
gzipped_filename = os.path.join(DATA_DIR, 'pds3_1band.IMG.gz')
bz2_filename = os.path.join(DATA_DIR, 'pds3_1band.IMG.bz2')


@pytest.fixture
def pattern_data():
    return numpy.loadtxt(os.path.join(DATA_DIR, 'pds3_1band.txt'))


def test_pds3_1band_labels():
    image = PDS3Image.open(filename)

    assert image.filename == filename
    assert image.bands == 1
    assert image.lines == 10
    assert image.samples == 10
    assert image.format == 'BAND_SEQUENTIAL'
    assert image.pixel_type == numpy.dtype('>i2')
    assert image.dtype == numpy.dtype('>i2')
    assert image.start_byte == 640
    assert image.shape == (1, 10, 10)
    assert image.byte_order == '>'
    assert image.size == 100
    assert image.compression == None


def test_gz_pds3_1band_labels():
    image = PDS3Image.open(gzipped_filename)
    assert image.filename == gzipped_filename
    assert image.bands == 1
    assert image.lines == 10
    assert image.samples == 10
    assert image.format == 'BAND_SEQUENTIAL'
    assert image.pixel_type == numpy.dtype('>i2')
    assert image.dtype == numpy.dtype('>i2')
    assert image.start_byte == 640
    assert image.shape == (1, 10, 10)
    assert image.byte_order == '>'
    assert image.size == 100
    assert image.compression == 'gz'


def test_bz2_pds3_1band_labels():
    image = PDS3Image.open(bz2_filename)
    assert image.filename == bz2_filename
    assert image.bands == 1
    assert image.lines == 10
    assert image.samples == 10
    assert image.format == 'BAND_SEQUENTIAL'
    assert image.pixel_type == numpy.dtype('>i2')
    assert image.dtype == numpy.dtype('>i2')
    assert image.start_byte == 640
    assert image.shape == (1, 10, 10)
    assert image.byte_order == '>'
    assert image.size == 100
    assert image.compression == 'bz2'


def test_image_save():
    image = PDS3Image.open(filename)
    image.save('Temp_Image.IMG')
    new_image = PDS3Image.open('Temp_Image.IMG')
    assert image.bands == new_image.bands
    assert image.lines == new_image.lines
    assert image.samples == new_image.samples
    assert image.format == new_image.format
    assert image.pixel_type == new_image.pixel_type
    assert image.dtype == new_image.dtype
    assert image.start_byte == new_image.start_byte
    assert image.shape == new_image.shape
    assert image.byte_order == new_image.byte_order
    assert image.size == new_image.size


def test_image_save_overwrite():
    image = PDS3Image.open(filename)
    image.save('Temp_Image.IMG')
    image_temp = PDS3Image.open('Temp_Image.IMG')
    image_temp.save(overwrite=True)
    new_image = PDS3Image.open(image_temp.filename)
    assert image_temp.filename == new_image.filename
    assert image_temp.bands == new_image.bands
    assert image_temp.lines == new_image.lines
    assert image_temp.samples == new_image.samples
    assert image_temp.format == new_image.format
    assert image_temp.pixel_type == new_image.pixel_type
    assert image_temp.dtype == new_image.dtype
    assert image_temp.start_byte == new_image.start_byte
    assert image_temp.shape == new_image.shape
    assert image_temp.byte_order == new_image.byte_order
    assert image_temp.size == new_image.size
    os.remove('Temp_Image.IMG')


def test_pds3_1band_image_format(pattern_data):
    image = PDS3Image.open(filename)

    assert image.data.shape == (10, 10)
    assert image.data.size == 100

    assert_almost_equal(image.data, pattern_data.reshape(10, 10))


def test_gz_pds3_1band_image_format(pattern_data):
    image = PDS3Image.open(gzipped_filename)
    assert image.data.shape == (10, 10)
    assert image.data.size == 100
    assert_almost_equal(image.data, pattern_data.reshape(10, 10))


def test_bz2_pds3_1band_image_format(pattern_data):
    image = PDS3Image.open(bz2_filename)
    assert image.data.shape == (10, 10)
    assert image.data.size == 100
    assert_almost_equal(image.data, pattern_data.reshape(10, 10))


def test_pds3_1band_disk_format(pattern_data):
    image = PDS3Image(open(filename, 'rb'), memory_layout='DISK')

    assert image.data.shape == (1, 10, 10)
    assert image.data.size == 100

    assert_almost_equal(image.data[0], pattern_data)


def test_gz_pds3_1band_disk_format(pattern_data):
    image = PDS3Image(gzip.open(gzipped_filename, 'rb'), compression='gz', memory_layout='DISK')
    assert image.data.shape == (1, 10, 10)
    assert image.data.size == 100
    assert_almost_equal(image.data[0], pattern_data)


def test_bz2_pds3_1band_disk_format(pattern_data):
    image = PDS3Image(bz2.BZ2File(bz2_filename, 'rb'), compression='bz2', memory_layout='DISK')
    assert image.data.shape == (1, 10, 10)
    assert image.data.size == 100
    assert_almost_equal(image.data[0], pattern_data)


def test_parse_pointer():
    # Example tests/mission_data/1p432690858esfc847p2111l2m1.img
    assert PDS3Image.parse_pointer(56, 640) == [35200, None]
    # Example tests/mission_data/W1782844276_1.LBL
    assert PDS3Image.parse_pointer(["W1782844276_1.IMG", 5], 1024) == [4096, 'W1782844276_1.IMG']
    # TODO: Awaiting other known valid examples to implement remaining conditions
    #print PDS3Image.parse_pointer("W1782844276_1.IMG", 1024)
    #print PDS3Image.parse_pointer(["W1782844276_1.IMG"], 1024)
    # ^IMAGE                         = 101337 <BYTES>
    assert PDS3Image.parse_pointer(Units(value=101337, units='BYTES'), None) == [101337, None]
