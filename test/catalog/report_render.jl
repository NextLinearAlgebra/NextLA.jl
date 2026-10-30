# Loaded by report.jl only when rendering, so summary tests need no plotting deps.
ENV["GKSwstype"] = "100"
using Plots
gr()
default(fontfamily="sans-serif", titlefontsize=15, guidefontsize=11, tickfontsize=9,
    legendfontsize=9, background_color=:white, foreground_color="#263648", dpi=180)

function metric_heatmap(d, title, subtitle, value; logarithm=log10, limits=nothing, legend="log10 value", diverging=false)
    values = [value(name, method) for name in d.matrices, method in d.methods]
    z = [isfinite(x) && x >= 0 ? logarithm(max(x,FLOOR)) : NaN for x in values]
    finite = filter(isfinite,vec(z))
    lo,hi = limits === nothing ? (isempty(finite) ? (-1.0,1.0) : extrema(finite)) : limits
    lo == hi && ((lo,hi)=(lo-.5,hi+.5))
    p = Plots.heatmap(1:length(d.methods),1:length(d.matrices),z;
        xticks=(1:length(d.methods),d.methods),yticks=(1:length(d.matrices),d.matrices),
        xrotation=40,yflip=true,clims=(lo,hi),colorbar_title=legend,
        color=diverging ? cgrad(["#B45419","#F7F7F7","#17659A"]) : cgrad(["#F1F5FA","#123F70"]),
        size=(1350,max(560,180+32length(d.matrices))),title=title,
        bottom_margin=16Plots.mm,left_margin=15Plots.mm,right_margin=10Plots.mm)
    for (i,name) in enumerate(d.matrices),(j,method) in enumerate(d.methods)
        isfinite(z[i,j]) && continue
        code=status(rowat(d,name,method))
        label=get(Dict("NOT_RECORDED"=>"...","OK"=>"—","TIMEOUT"=>"T/O","ERROR"=>"ERR",
            "NONFINITE"=>"NF","SKIP_MEMORY"=>"MEM","MISSING"=>"MISS"),code,code)
        annotate!(p,j,i,text(label,8,"#4D5260"))
    end
    p
end

function scatterplot(d, title, subtitle, points; xlabel, ylabel, xlog=true, ylog=true,
        horizontal=nothing, diagonal=false, lines=false, xrange=nothing, yrange=nothing)
    valid=filter(p -> isfinite(p.x) && isfinite(p.y) && (!xlog || p.x>=0) && (!ylog || p.y>=0),points)
    transform(v,log)=log ? log10(max(v,FLOOR)) : v
    function bounds(values,range,log)
        range === nothing || return range
        isempty(values) && return log ? (1e-8,1.0) : (0.0,1.0)
        lo,hi=extrema(transform(v,log) for v in values)
        pad=lo==hi ? .5 : .08*(hi-lo)
        log ? (10.0^(lo-pad),10.0^(hi+pad)) : (lo-pad,hi+pad)
    end
    xl=bounds([p.x for p in valid],xrange,xlog)
    yl=bounds(vcat([p.y for p in valid],horizontal===nothing ? Float64[] : [horizontal]),yrange,ylog)
    p=plot(;title=title,xlabel=xlabel,ylabel=ylabel,xscale=xlog ? :log10 : :identity,
        yscale=ylog ? :log10 : :identity,xlims=xl,ylims=yl,legend=:outerright,
        size=(1250,650),gridalpha=.2,left_margin=8Plots.mm,bottom_margin=8Plots.mm)
    horizontal === nothing || hline!(p,[horizontal];label="equal runtime",color=:gray,linestyle=:dash)
    if diagonal
        lo,hi=max(xl[1],yl[1]),min(xl[2],yl[2])
        lo<hi && plot!(p,[lo,hi],[lo,hi];label="equal error",color=:gray,linestyle=:dash)
    end
    for (j,method) in enumerate(d.methods)
        selected=filter(p -> p.method==method,valid)
        isempty(selected) && continue
        lines && sort!(selected;by=p -> p.x)
        xs=[xlog ? max(s.x,FLOOR) : s.x for s in selected]
        ys=[ylog ? max(s.y,FLOOR) : s.y for s in selected]
        color=COLORS[mod1(j,length(COLORS))]
        if lines
            plot!(p,xs,ys;label=method,color=color,marker=:circle,linewidth=2,markersize=4)
        else
            scatter!(p,xs,ys;label=method,color=color,markersize=5,markeralpha=.8,markerstrokewidth=.5)
        end
    end
    if isempty(valid)
        x=xlog ? sqrt(xl[1]*xl[2]) : mean(xl)
        y=ylog ? sqrt(yl[1]*yl[2]) : mean(yl)
        annotate!(p,x,y,text("No eligible observations in this snapshot",11,:gray))
    end
    p
end

function statusplot(d)
    states=sort(unique(vcat([status(r) for r in values(d.lookup)],["NOT_RECORDED"])))
    counts=[count(name -> status(rowat(d,name,method))==state,d.matrices) for method in d.methods,state in states]
    bar(1:length(d.methods),counts;bar_position=:stack,label=permutedims(states),
        color=permutedims([get(STATUS_COLORS,s,"#555555") for s in states]),
        xticks=(1:length(d.methods),d.methods),xrotation=35,ylims=(0,length(d.matrices)),
        ylabel="Selected matrices",title="Completion and failures (OK is not an accuracy pass)",
        legend=:outerright,size=(1250,600),bottom_margin=13Plots.mm,linecolor=:white)
end
function stilesplot(d)
    selected=[r for name in d.matrices for r in [rowat(d,name,"sTiles_F64")]
        if ok(r,"analysis_ms") && ok(r,"factor_ms") && num(r,"analysis_ms")+num(r,"factor_ms")>0]
    isempty(selected) && return plot(;title="sTiles phase costs: no completed measurements",axis=false,grid=false,legend=false)
    analysis=[100num(r,"analysis_ms")/(num(r,"analysis_ms")+num(r,"factor_ms")) for r in selected]
    bar(1:length(selected),hcat(analysis,100 .- analysis);orientation=:h,bar_position=:stack,
        yticks=(1:length(selected),getindex.(selected,"matrix")),yflip=true,xlims=(0,100),
        label=["Analysis" "Numeric factorization"],color=["#56B4E9" "#0072B2"],linecolor=:white,
        xlabel="Share of minimum combined analysis + factor time (%)",title="sTiles phase costs",
        legend=:outertopright,size=(1250,max(480,150+30length(selected))),left_margin=16Plots.mm)
end
