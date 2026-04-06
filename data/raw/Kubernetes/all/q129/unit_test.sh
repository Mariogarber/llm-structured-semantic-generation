kubectl apply -f labeled_code.yaml

sleep 10
if [[ $(kubectl get pv nfs-backup-volume -o=jsonpath='{.status.phase}') != "Bound" ]]; then
    kubectl delete -f labeled_code.yaml
    exit 1
fi
sleep 5
# Check the bound PV for the PVC
bound_pv=$(kubectl get pvc nfs-backup-volume-claim -o=jsonpath='{.spec.volumeName}')
echo $bound_pv

if [ "$bound_pv" == "nfs-backup-volume" ]; then
    echo "PVC is bound to the correct PV."
else
    kubectl delete -f labeled_code.yaml
    exit 1
fi

# Check the details of the PV
desc=$(kubectl describe pv "$bound_pv")
if [[ $desc == *"/my/nfs/"* ]] && 
   [[ $desc == *"StorageClass:    manual"* ]] &&
   [[ $desc == *"1Gi"* ]] &&
   [[ $desc == *"NFS"* ]]; then
    echo cloudeval_unit_test_passed
else
    echo "Test failed"
fi
kubectl delete -f labeled_code.yaml