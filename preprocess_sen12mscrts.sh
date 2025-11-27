#!/bin/bash

# download ERA5 for SEN12MSCRTS
eurocropsssl-cli datasets era5 download
# preprocess SEN12MS-CR-TS
eurocropsssl-cli datasets sen12mscrts preprocess