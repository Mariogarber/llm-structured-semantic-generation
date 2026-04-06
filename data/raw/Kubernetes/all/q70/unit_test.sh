kubectl apply -f labeled_code.yaml
kubectl wait deploy --all --for=condition=available --timeout=30s

if kubectl get deploy -o jsonpath='{.items[*].metadata.name})' | grep "my-deploy"; then
    if kubectl get deploy my-deploy -o jsonpath='{.spec.replicas}' | grep "3"; then
        if kubectl get deploy my-deploy -o jsonpath='{.spec.template.spec.containers[*].image}'| grep "nginx:1.16.1"; then
            if kubectl get deploy my-deploy -o jsonpath='{.spec.template.spec.containers[*].ports[*].containerPort}'| grep  "8080"; then
                echo cloudeval_unit_test_passed
            fi
        fi
    fi
fi
