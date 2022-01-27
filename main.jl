# examples/01-hello.jl
using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

VELOCITYFACTOR = 10
GRIDSIZE = 35

struct Cell
    cellid::Tuple{Int, Int}
    particlepositions::Array{Float32,2}
    particletypes::Array{Int32,1}
end

function moveparticles!(cell::Cell)

    # The particles move using the cos of their position as velocity
    dp = cos.(cell.particlepositions)

    # Scale the velocity by the velocity factor
    dp = dp ./ VELOCITYFACTOR

    # The velocity is affected by the type of the particle
    typefactor = map(x->if x==1 1 elseif x==2 2 else 5 end, cell.particletypes)
    typefactor = vcat(typefactor,typefactor)
    dp = dp ./ typefactor

    # Move the particles
    cel.particlepositions = cell.particlepositions .+ dp
end

function computetransfers!(cell::Cell)

    # Round down to determine the cell to send to
    corresponding = round(Int32,cell.particlepositions)
    # Wrap around
    map!(v->if v < 1 GRIDSIZE elseif v > GRIDSIZE 1 else v end, corresponding)

    postransfers = Dict{Tuple{Int32,Int32},Vector{Array{Float32,1}}}()
    typetransfers = Dict{Tuple{Int32,Int32},Vector{Array{Int32,1}}}()

    newpositions = Vector{Array{Float32,1}}()
    newtypes = Vector{Int32,1}()

    for i in 1:size(corresponding,1)
        (x,y) = corresponding[i,:]
        if x != cell.cellid[1] || y != cell.cellid[2]
            # Send the particle to the corresponding cell
            if !haskey(postransfers,(x,y))
                postransfers[(x,y)] = Vector{Array{Float32,1}}()
                typetransfers[(x,y)] = Vector{Array{Int32,1}}()
            end
            push!(postransfers[(x,y)],cell.particlepositions[i,:])
            push!(typetransfers[(x,y)],cell.particletypes[i])
        else 
            # Keep the particle in the current cell
            push!(newpositions, cell.particlepositions[i,:])
            push!(newtypes, cell.particletypes[i])
        end
    end

    cell.particlepositions = Array(hcat(newpositions...)')
    cell.particletypes = newtypes

    return postransfers, typetransfers
end

function tickcells!(cells::Array{Cell},
                    ranktocells::Vector{Set{Tuple{Int32,Int32}}})
    # Move particles
    for cell in cells
        moveparticles!(cell)
    end

    # Compute transfers
    postransfers = Dict{Tuple{Int32,Int32},Vector{Array{Float32,1}}}()
    typetransfers = Dict{Tuple{Int32,Int32},Vector{Array{Int32,1}}}()

    for cell in cells
        localptransfers, localttransfers = computetransfers!(cell)
        # Merge the transfers
        for (k,v) in localptransfers
            if !haskey(postransfers,k)
                postransfers[k] = Vector{Array{Float32,1}}()
                typetransfers[k] = Vector{Array{Int32,1}}()
            end
            postransfers[k] = vcat(postransfers[k],v)
            typetransfers[k] = vcat(typetransfers[k],localttransfers[k])
        end
    end

    # Communicate sizes of transfers to other ranks
    
end


println("Hello world, I am $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")
MPI.Barrier(comm)

if MPI.Comm_rank(comm) == 0 
    send = [Particle(52,57,42)]
    handle = MPI.Isend(send, 1, 32, comm)
    MPI.Waitall!([handle])
    println("Sent")
else
    recv = Array{Particle}(undef,1)
    handle = MPI.Irecv!(recv,0,32,comm)
    MPI.Waitall!([handle])
    println("Received: ", recv[1])
end

MPI.Barrier(comm)