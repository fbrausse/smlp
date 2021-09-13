# export BIOS MRX TX/RX training datasets from Shai.
export_bios_mrc_tx_rx_datasets = function(input_df, export_mrc_data, product, rank, divide=0) {
   #pnvv(colnames(input_df)); pnv(export_mrc_data)

   if (product == 'ICL') {
      tx_colnms = c("Timing","Area","CH","RANK","Byte","TXEQ_DEEM","TXEQ_GAIN","CPU_RON_UP",
          "LP4_DIMM_ODT_WR","delta")
      rx_colnms = c("Timing","Area","CH","RANK","Byte","LP4_DIMM_RON","CPU_ODT_UP","LP4_SOC_ODT",
          "ICOMP","CTLE_C","CTLE_R","delta")
   } else if (product == "RKL") {
      tx_colnms = c("Timing","Area","CH","RANK","Byte","TXEQ_DEEM","TXEQ_GAIN", "DDR4_RTT_NOM.1",
                            "DDR4_RTT_PARK.1", "DDR4_RTT_WR", "CPU_RON_UP", "delta")
      # just drop the ".1" suffix in two duplicate columns "DDR4_RTT_NOM.1" "DDR4_RTT_PARK.1"
      tx_colnms_fixed = c("Timing","Area","CH","RANK","Byte","TXEQ_DEEM","TXEQ_GAIN", "DDR4_RTT_NOM",
                            "DDR4_RTT_PARK", "DDR4_RTT_WR", "CPU_RON_UP", "delta")
      rx_colnms = c("Timing","Area","CH","RANK","Byte", 'DDR4_DIMM_RON', "DDR4_RTT_NOM",
                            "DDR4_RTT_PARK", "CPU_ODT_UP","ICOMP","CTLE_C","CTLE_R", "delta")
   } else if (product == 'ADLS') { 
      # Area is available in TX and not in RX; not required anyway, dropping from both
      # delta is available in TX and not in RX; recomuting for TX as well as Up - Down
      tx_colnms = c("Timing","MC","RANK","Byte", "DDR4_RTT_NOM_TX", "DDR4_RTT_PARK_TX", "DDR4_RTT_WR",
                            "TXEQ_COEFF0", "TXEQ_COEFF1", "TXEQ_COEFF2", "CPU_RON", "delta") #"CPU_RON_UP
      # 'DDR4_DIMM_RON', not toggled in recent dataset; might need to add agin to rx_colnms later
      # also in the recent datset CPU_ODT_UP ->CPU_ODT 
      rx_colnms = c("Timing","MC","RANK","Byte", "DDR4_RTT_NOM_RX",
                            "DDR4_RTT_PARK_RX", "CPU_ODT","CTLE_EQ","CTLE_C","CTLE_R",
                            "RXBIAS_CTL", "RXBIAS_TAIL_CTL", "RXBIAS_VREFSEL", "delta")             
   } else {
      eva_stop(str_vec="Unsupported product in export_bios_mrc_tx_rx_datasets")
   }

   # derive/extract the required data
   #if (product == "RKL" | product == "ICL") {
      # for ADLS we use the delta column that already exists in the training data
   input_df$delta = round(input_df$Up - input_df$Down, 4);  #pnv(input_df$delta);
   #}
   colnms = colnames(input_df)
   if (export_mrc_data == "tx") {
      pnvv(colnms); pnvv(setdiff(tx_colnms, colnms));  pnvv(setdiff(colnms, tx_colnms)); 
      sub_df = input_df[ tx_colnms ]
      #pnv(colnames(sub_df)); pnv(tx_colnms_fixed)
      if (product == "RKL") {
            colnames(sub_df) = tx_colnms_fixed
      }
   } else if (export_mrc_data == "rx") {
      pnvv(colnms); pnvv(setdiff(rx_colnms, colnms));  pnvv(setdiff(colnms, rx_colnms)); 
      sub_df = input_df[ rx_colnms ]
   } else eva_stop(str_vec="Incorrect data type in function export_bios_mrc_tx_rx_datasets")
   #snv(sub_df)
   # write out the required data and return
   data_name = paste(c(export_mrc_data, "csv"),  collapse="."); #pnv(data_name)
   #pnv(output_directory); pnv(output_file_base_name)
   output_file_name = paste(c(output_directory, data_name), collapse="/"); #pnv(output_file_name)

   if (!is.null(rank)) {
      sub_df = sub_df[sub_df$RANK == rank, ]
      sub_df$RANK <- NULL
   }
   if (divide == 0) {
      eva_write_csv(sub_df, output_file_name, F, "none"); q()
   } else {
     rows_count = ceiling(nrow(sub_df) / divide)
     for (i in 1:divide) {
            row_start = (i-1)*rows_count + 1
            row_end = min(i*rows_count, nrow(sub_df))
            print(c("dumping rows", row_start, row_end))
            curr_df = sub_df[row_start:row_end, ]
            eva_write_csv(curr_df, paste(c(output_file_name, i), collapse = "_"), F, "none"); 
     }
     q()
   }
}

Usage: (run one at a time)
       export_bios_mrc_tx_rx_datasets(input_df, "tx", "ADLS", 0, 0); q()
       export_bios_mrc_tx_rx_datasets(input_df, "rx", "ADLS", 0, 0); q()


