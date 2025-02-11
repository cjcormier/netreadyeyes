import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
HIGH_RES_DIR = os.path.join(BASE_DIR, "high_res_images")
LOW_RES_DIR = os.path.join(BASE_DIR, "low_res_images")
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "test_images")
INSTALLED_APPS = [
    'bootstrap4',
    'crispy_forms',
    'image_matcher.apps.ImageMatcherConfig',
    'rest_framework',
]

# LANGUAGE_CODE = 'en-us'
# TIME_ZONE = 'UTC'
# USE_I18N = True
# USE_L10N = True
# USE_TZ = True

CRISPY_TEMPLATE_PACK = 'bootstrap3'
