#!/bin/bash

# REPLACE WITH YOUR WATID
WATID=h3trinh

# SHELL TO CHANGE
NEWSHELL=zsh

# LIST OF TESLA NODES
COMPUTE_NODE_LIST=("ecetesla0" "ecetesla1" "ecetesla2" "ecetesla3" "ecetesla4")

# Loop through each node and change shell
change_cn_shell(){
    for CN in "${COMPUTE_NODE_LIST[@]}"; do
        echo "------------------------------------"
        echo "Changing shell to $NEWSHELL on $CN..."
        ssh $WATID@$CN "chsh -s /bin/$NEWSHELL"
        echo "Finish changing shell to $NEWSHELL on $CN completed."
    done
}

validate_cn_shell(){
    for CN in "${COMPUTE_NODE_LIST[@]}"; do
      CN_SHELL=$(ssh $WATID@$CN "echo \$SHELL")

      if [[ "$CN_SHELL" != "/bin/$NEWSHELL" ]];then
          echo "Error: Shell on $CN is $CN_SHELL but expected /bin/$NEWSHELL"
          continue
      fi

      echo "Shell on $CN is correctly set to $CN_SHELL"
    done
}

# Call change shells to compute nodes
change_cn_shell

# Validate new updated shells of compute nodes
validate_cn_shell
