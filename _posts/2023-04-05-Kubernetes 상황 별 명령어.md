---
layout: article
title:  "Kubernetes 상황 별 명령어"
category: [MLops]
tag: [pod, Kubernetes, MLops]
permalink: /KubernetesCommand/
show_author_profile: true
aside:
    toc: true
sidebar:
    nav: "study-nav"
---

> 본 포스터는 패스트캠퍼스 [머신러닝 서비스 구축을 위한 실전 MLOps](https://fastcampus.co.kr/data_online_mlops)를 참고했습니다.

# Pod

## 생성, 조회, 삭제

```bash
# pod 생성
kubectl apply -f pod.yaml 

# pod 조회
kubectl get pod

# kube-system 이라는 namespace를 가진 pod 조회
kubectl get pod -n kube-system

# 모든 pod 조회
kubectl get pod -A

# 하나의 pod 조회
kubectl get pod <pod-name>

# 하나의 pod 자세히 조회
kubectl describe pod <pod-name>

# pod 삭제. pod-name으로 삭제하거나 실행했던 yaml 파일로 삭제도 가능.
kubectl delete pod <pod-name>
kubectl delete -f <YAML-파일-경로>
```

## 로그 확인

```bash
# pod의 로그 계속 조회. 한번만 조회할거면 -f 옵션 빼기
kubectl logs <pod-name> -f

# pod 안에 여러개 container가 있을 경우
kubectl logs <pod-name> -c <container-name> -f
```


## 내부 접속

```bash
kubectl exec -it <pod-name> -- <명령어>

# pod 안에 여러개 container가 있을 경우
kubectl exec -it <pod-name> -c <container-name> -- <명령어>
```